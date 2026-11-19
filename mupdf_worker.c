/*
    KOReader: asynchronous MuPDF rasterization worker
    Copyright (C) 2026 KOReader contributors

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
    Background rasterization worker for MuPDF display lists.

    MuPDF is not thread-safe for document/page access, but it explicitly
    supports a two-stage pipeline:
      1. Main thread: fz_new_display_list_from_page() records an immutable
         display list (parsing only, no rasterization).
      2. Worker thread: fz_run_display_list() on a cloned context
         (fz_clone_context) rasterizes it into a caller-provided buffer.

    Shared resources (the store, the glyph cache, FreeType) are protected
    by client-provided locks. The main context is created with pthread
    locks right away by wrap-mupdf.c's mupdf_new_context(), so contexts
    (and clones) are always safe for concurrent use, following the
    pattern of MuPDF's own multi-threaded example.

    Contract:
      - mupdf_worker_init() must be called once, from the main thread,
        before any concurrent MuPDF use.
      - The display list handed to mupdf_worker_submit() must stay alive
        until the job leaves the RUNNING state; submit takes its own
        reference and drops it when the job finishes.
      - The destination buffer is owned by the caller, which is
        responsible for keeping it alive (and unreferenced by anyone else)
        until the job finishes.
      - Only one job may be in flight; a submit while RUNNING returns busy.
*/

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mupdf/fitz.h>

/* Symbol visibility, matching wrap-mupdf.h on non-Windows builds. */
#define WORKER_PUBLIC __attribute__((visibility("default")))

typedef enum {
    WORKER_IDLE = 0,
    WORKER_RUNNING,
    WORKER_DONE,
    WORKER_ERROR,
    WORKER_ABORTED,
} WorkerState;

/* Colorspace selector mirroring ffi/mupdf.lua's draw_new() choices. */
enum {
    WORKER_CS_GRAY = 0,
    WORKER_CS_RGB = 1,
    WORKER_CS_BGR = 2,
};

typedef struct {
    fz_display_list *list;
    fz_matrix ctm;
    fz_rect scissor;
    fz_irect bbox;
    unsigned char *bb_data;
    int stride;
    int colorspace;
    int alpha;
    int transparency_mask;
    float gamma;
    float saturation;
    fz_cookie cookie;
    WorkerState state;
} WorkerJob;

typedef struct {
    fz_context *main_ctx;
    fz_context *worker_ctx;
    WorkerJob job;

    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    int shutdown;
    int initialized;
} MuPdfWorker;

static MuPdfWorker g_worker = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_start = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

/*
    Client-provided lock functions for the shared MuPDF context live in
    wrap-mupdf.c (mupdf_new_context), which creates the root context with
    them from the start; clones made here inherit the very same locks.
*/

/*
    Transparency mask passthrough device, replicating wrap-mupdf.c's
    new_transparency_mask_device(): pages with soft masks must not have
    their background clobbered by the explicit clear, so we forward only
    image-mask fills to the real draw device. The page-level *check*
    (page_has_transparency_mask) happens on the main thread at submit
    time; this device only touches the draw device stack, which is
    worker-local.
*/
static void transparency_mask_close_device(fz_context *ctx, fz_device *dev)
{
    fz_close_device(ctx, dev->passthrough);
}

static void transparency_mask_drop_device(fz_context *ctx, fz_device *dev)
{
    fz_drop_device(ctx, dev->passthrough);
}

static void transparency_mask_fill_image(fz_context *ctx, fz_device *dev, fz_image *img, fz_matrix ctm, float alpha, fz_color_params color_params)
{
    if (img->mask) {
        float black = 0.f;
        fz_fill_image_mask(ctx, dev->passthrough, img->mask, ctm, fz_device_gray(ctx), &black, alpha, color_params);
    }
}

static fz_device *new_transparency_mask_device(fz_context *ctx, fz_device *dev)
{
    fz_device *transparency_mask_dev = fz_new_derived_device(ctx, fz_device);
    transparency_mask_dev->passthrough = fz_keep_device(ctx, dev);
    transparency_mask_dev->close_device = transparency_mask_close_device;
    transparency_mask_dev->drop_device = transparency_mask_drop_device;
    transparency_mask_dev->fill_image = transparency_mask_fill_image;
    return transparency_mask_dev;
}

static fz_colorspace *worker_colorspace(fz_context *ctx, int colorspace)
{
    switch (colorspace) {
        case WORKER_CS_RGB:
            return fz_device_rgb(ctx);
        case WORKER_CS_BGR:
            return fz_device_bgr(ctx);
        default:
            return fz_device_gray(ctx);
    }
}

/*
    Post-processing steps normally done by the caller right after drawing
    (c.f., page_mt.draw_new in ffi/mupdf.lua): gamma correction through
    fz_gamma_pixmap, then luma-preserving saturation. Running them here
    keeps arbitrarily large buffers off the UI thread.
*/

static inline uint8_t worker_saturate_channel(uint8_t channel, uint8_t gray, double saturation)
{
    double value = (double)gray + (((double)channel - (double)gray) * saturation);
    if (value <= 0.0) {
        return 0;
    }
    if (value >= 255.0) {
        return 255;
    }
    return (uint8_t)(value + 0.5);
}

/* Replicates blitbuffer.c's BB_saturate_rect BBRGB32 path on the pixmap's
 * raw samples (grayscale and monochrome buffers, n < 3, have no saturation).
 * Checks cookie->abort every 64 scanlines so an obsolete job is abandoned
 * promptly instead of finishing a doomed full-buffer pass. */
static void worker_saturate_pixmap(fz_context *ctx, fz_pixmap *pix, double saturation, fz_cookie *cookie)
{
    if (pix->n < 3) {
        return;
    }
    /* Component order follows the colorspace: RGB gives r,g,b; BGR gives
     * b,g,r — green is always the second sample. */
    const int r_off = pix->colorspace == fz_device_bgr(ctx) ? 2 : 0;
    const int b_off = 2 - r_off;
    for (int y = 0; y < pix->h; y++) {
        if ((y & 63) == 0 && cookie->abort) {
            return;
        }
        unsigned char *row = pix->samples + (size_t)y * pix->stride;
        for (int x = 0; x < pix->w; x++) {
            unsigned char *px = row + (size_t)x * pix->n;
            const uint8_t r = px[r_off];
            const uint8_t g = px[1];
            const uint8_t b = px[b_off];
            const uint8_t gray = (uint8_t)((4898U * r + 9618U * g + 1869U * b) >> 14U);
            px[r_off] = worker_saturate_channel(r, gray, saturation);
            px[1] = worker_saturate_channel(g, gray, saturation);
            px[b_off] = worker_saturate_channel(b, gray, saturation);
        }
    }
}

/* Row-wise replica of fz_gamma_pixmap (source/fitz/pixmap.c): a 256-entry
 * LUT applied to every non-alpha component. Reimplemented locally so the
 * scan can check cookie->abort every 64 scanlines like the saturation pass
 * (fz_gamma_pixmap itself is monolithic and not abortable). */
static void worker_gamma_pixmap(fz_pixmap *pix, float gamma, fz_cookie *cookie)
{
    unsigned char gamma_map[256];
    const int n1 = pix->n - pix->alpha;
    const int n = pix->n;

    for (int k = 0; k < 256; k++) {
        gamma_map[k] = powf(k / 255.0f, gamma) * 255;
    }

    for (int y = 0; y < pix->h; y++) {
        if ((y & 63) == 0 && cookie->abort) {
            return;
        }
        unsigned char *s = pix->samples + (size_t)y * pix->stride;
        for (int x = 0; x < pix->w; x++) {
            for (int k = 0; k < n1; k++) {
                s[k] = gamma_map[s[k]];
            }
            s += n;
        }
    }
}

/* cookie points at the SHARED in-flight job's cookie, so aborts requested
 * by the main thread are honored mid-job (the local job copy below is not
 * updated by mupdf_worker_abort). */
static void worker_render_job(fz_context *ctx, WorkerJob *job, fz_cookie *cookie)
{
    fz_colorspace *cs = worker_colorspace(ctx, job->colorspace);
    fz_pixmap *pix = NULL;
    fz_device *dev = NULL;

    fz_try(ctx) {
        pix = fz_new_pixmap_with_bbox_and_data(ctx, cs, job->bbox, NULL, job->alpha, job->bb_data);
        fz_clear_pixmap_with_value(ctx, pix, 0xff);
        /* The device transform MUST stay identity: the raster matrix travels
         * ONLY via fz_run_display_list's top_ctm (the sync path, run_page in
         * ffi/mupdf.lua, does the same with mupdf_new_draw_device(ctx, nil, ...)
         * + fz_run_page(ctx, page, dev, ctm)). Passing job->ctm here as well
         * double-applies it (the draw device concatenates dev->transform onto
         * every op, c.f., fz_draw_fill_path_aux), throwing content off the
         * pixmap for any job with a non-zero bbox origin — e.g., a viewport
         * excerpt tile whose clip region starts at (size_x, size_y): the
         * content lands at native*zoom² and is clipped out entirely (blank
         * tile), or lands shifted for larger excerpts. */
        dev = fz_new_draw_device(ctx, fz_identity, pix);

        if (job->transparency_mask) {
            fz_device *mask_dev = new_transparency_mask_device(ctx, dev);
            fz_drop_device(ctx, dev); /* mask_dev holds the reference now. */
            dev = mask_dev;
        }

        fz_run_display_list(ctx, job->list, dev, job->ctm, job->scissor, cookie);

        fz_close_device(ctx, dev);
        fz_drop_device(ctx, dev);
        dev = NULL;

        if (job->gamma >= 0.0f && fabsf(job->gamma - 1.0f) >= 1e-4f) {
            worker_gamma_pixmap(pix, job->gamma, cookie);
        }
        if (fabsf(job->saturation - 1.0f) >= 1e-4f) {
            worker_saturate_pixmap(ctx, pix, job->saturation, cookie);
        }
    }
    fz_always(ctx) {
        fz_drop_pixmap(ctx, pix);
    }
    fz_catch(ctx) {
        if (dev) {
            fz_drop_device(ctx, dev);
        }
        fz_rethrow(ctx);
    }
}

static void *worker_thread_func(void *arg)
{
    MuPdfWorker *w = (MuPdfWorker *)arg;

    while (1) {
        pthread_mutex_lock(&w->mutex);
        while (w->job.state != WORKER_RUNNING && !w->shutdown) {
            pthread_cond_wait(&w->cond_start, &w->mutex);
        }
        if (w->shutdown) {
            pthread_mutex_unlock(&w->mutex);
            break;
        }

        WorkerJob job = w->job;
        /* The render watches the SHARED job's cookie (mupdf_worker_abort()
         * writes it under the mutex), so cancellation lands at the next
         * progress step or scanline checkpoint. Reading the plain flag
         * without holding the mutex mirrors MuPDF's own cookie usage in
         * its threaded examples. */
        fz_cookie *cookie = &w->job.cookie;
        pthread_mutex_unlock(&w->mutex);

        /* TEST KNOB (device-latency replication): optional per-job delay so
         * desktop runs can exercise the slow-worker paths (coalescing
         * timeouts, pending-hint parking, poll races) that real devices
         * exhibit with multi-second jobs on a downclocked background core.
         * Sleeps WITHOUT holding the mutex, before any rendering: the job
         * simply takes that long, exactly like the device. The sleep is
         * sliced so a cooperative abort (mupdf_worker_abort) lands during
         * it, like it would during a long real render; the render call
         * below then bails out at its first checkpoint. */
        {
            const char *delay_env = getenv("MUPDF_WORKER_DELAY_MS");
            if (delay_env != NULL) {
                long delay_ms = atol(delay_env);
                while (delay_ms > 0 && !w->job.cookie.abort) {
                    long slice_ms = delay_ms > 20 ? 20 : delay_ms;
                    struct timespec ts = {
                        slice_ms / 1000,
                        (slice_ms % 1000) * 1000000L,
                    };
                    nanosleep(&ts, NULL);
                    delay_ms -= slice_ms;
                }
            }
        }

        WorkerState result = WORKER_DONE;
        fz_try(w->worker_ctx) {
            worker_render_job(w->worker_ctx, &job, cookie);
            if (cookie->abort) {
                result = WORKER_ABORTED;
            }
        }
        fz_catch(w->worker_ctx) {
            result = cookie->abort ? WORKER_ABORTED : WORKER_ERROR;
        }

        /* Release the reference taken at submit time using the worker's own
         * cloned context (fz_context is strictly single-threaded in MuPDF;
         * passing main_ctx here races with main-thread MuPDF calls). */
        fz_drop_display_list(w->worker_ctx, job.list);

        pthread_mutex_lock(&w->mutex);
        if (w->job.state == WORKER_RUNNING) {
            w->job.state = result;
        }
        pthread_cond_broadcast(&w->cond_done);
        pthread_mutex_unlock(&w->mutex);
    }

    return NULL;
}

WORKER_PUBLIC int mupdf_worker_init(fz_context *main_ctx)
{
    pthread_mutex_lock(&g_worker.mutex);
    if (g_worker.initialized) {
        pthread_mutex_unlock(&g_worker.mutex);
        return 0;
    }

    /* The main context already carries real pthread locks (installed at
     * creation time by mupdf_new_context), so it can be cloned. */
    g_worker.main_ctx = main_ctx;
    g_worker.worker_ctx = fz_clone_context(main_ctx);
    if (g_worker.worker_ctx == NULL) {
        fprintf(stderr, "mupdf_worker: fz_clone_context failed\n");
        pthread_mutex_unlock(&g_worker.mutex);
        return -1;
    }

    g_worker.job.state = WORKER_IDLE;
    g_worker.shutdown = 0;

    if (pthread_create(&g_worker.thread, NULL, worker_thread_func, &g_worker) != 0) {
        fprintf(stderr, "mupdf_worker: pthread_create failed\n");
        fz_drop_context(g_worker.worker_ctx);
        g_worker.worker_ctx = NULL;
        pthread_mutex_unlock(&g_worker.mutex);
        return -1;
    }

    g_worker.initialized = 1;
    pthread_mutex_unlock(&g_worker.mutex);
    return 0;
}

WORKER_PUBLIC int mupdf_worker_submit(fz_display_list *list, const fz_matrix *ctm, const fz_rect *scissor, const fz_irect *bbox, unsigned char *bb_data, int stride, int colorspace, int alpha, int transparency_mask, float gamma, float saturation)
{
    pthread_mutex_lock(&g_worker.mutex);
    if (!g_worker.initialized || g_worker.job.state == WORKER_RUNNING) {
        pthread_mutex_unlock(&g_worker.mutex);
        return -1; /* Not initialized or busy. */
    }

    g_worker.job.list = fz_keep_display_list(g_worker.main_ctx, list);
    g_worker.job.ctm = *ctm;
    g_worker.job.scissor = *scissor;
    g_worker.job.bbox = *bbox;
    g_worker.job.bb_data = bb_data;
    g_worker.job.stride = stride;
    g_worker.job.colorspace = colorspace;
    g_worker.job.alpha = alpha;
    g_worker.job.transparency_mask = transparency_mask;
    g_worker.job.gamma = gamma;
    g_worker.job.saturation = saturation;
    g_worker.job.cookie.abort = 0;
    g_worker.job.state = WORKER_RUNNING;

    pthread_cond_signal(&g_worker.cond_start);
    pthread_mutex_unlock(&g_worker.mutex);
    return 0;
}

WORKER_PUBLIC int mupdf_worker_status(void)
{
    pthread_mutex_lock(&g_worker.mutex);
    int state = g_worker.job.state;
    pthread_mutex_unlock(&g_worker.mutex);
    return state;
}

WORKER_PUBLIC void mupdf_worker_abort(void)
{
    pthread_mutex_lock(&g_worker.mutex);
    if (g_worker.job.state == WORKER_RUNNING) {
        g_worker.job.cookie.abort = 1;
    }
    pthread_mutex_unlock(&g_worker.mutex);
}

/*
    Wait for the in-flight job to leave the RUNNING state.
    timeout_ms < 0 waits indefinitely; otherwise it is a bounded wait in
    milliseconds. Returns the resulting state.
*/
WORKER_PUBLIC int mupdf_worker_wait(int timeout_ms)
{
    pthread_mutex_lock(&g_worker.mutex);
    if (g_worker.job.state == WORKER_RUNNING) {
        if (timeout_ms < 0) {
            while (g_worker.job.state == WORKER_RUNNING) {
                pthread_cond_wait(&g_worker.cond_done, &g_worker.mutex);
            }
        } else {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += (long)timeout_ms * 1000000L;
            while (ts.tv_nsec >= 1000000000L) {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000L;
            }
            pthread_cond_timedwait(&g_worker.cond_done, &g_worker.mutex, &ts);
        }
    }
    int state = g_worker.job.state;
    pthread_mutex_unlock(&g_worker.mutex);
    return state;
}

WORKER_PUBLIC void mupdf_worker_shutdown(void)
{
    pthread_mutex_lock(&g_worker.mutex);
    if (!g_worker.initialized) {
        pthread_mutex_unlock(&g_worker.mutex);
        return;
    }
    g_worker.shutdown = 1;
    g_worker.job.cookie.abort = 1;
    pthread_cond_signal(&g_worker.cond_start);
    pthread_mutex_unlock(&g_worker.mutex);

    pthread_join(g_worker.thread, NULL);

    pthread_mutex_lock(&g_worker.mutex);
    g_worker.initialized = 0;
    if (g_worker.worker_ctx) {
        fz_drop_context(g_worker.worker_ctx);
        g_worker.worker_ctx = NULL;
    }
    pthread_mutex_unlock(&g_worker.mutex);
}
