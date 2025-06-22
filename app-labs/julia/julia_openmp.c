#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 19800
#define HEIGHT 19800
#define MAX_ITER 10000
#define CHUNK_HEIGHT 1980

int julia(double x, double y, double cx, double cy) {
    int i;
    for (i = 0; i < MAX_ITER; i++) {
        double xtemp = x * x - y * y + cx;
        y = 2 * x * y + cy;
        x = xtemp;
        if (x * x + y * y > 4.0) break;
    }
    return i;
}

void generate_reference_image(const char *filename) {
    int **image = malloc(sizeof(int *) * HEIGHT);
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = malloc(sizeof(int) * WIDTH);
    }

    double cx = -0.7, cy = 0.27015;
    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = 1.5 * (i - WIDTH / 2) / (0.5 * WIDTH);
            double y = (j - HEIGHT / 2) / (0.5 * HEIGHT);
            image[j][i] = julia(x, y, cx, cy);
        }
    }

    FILE *img = fopen(filename, "w");
    if (img) {
        fprintf(img, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                if (j % CHUNK_HEIGHT == 0) {
                    fprintf(img, "255 0 0 ");
                } else {
                    int iter = image[j][i];
                    int color = (int)(255.0 * iter / MAX_ITER);
                    fprintf(img, "%d %d %d ", color, color, color);
                }
            }
            fprintf(img, "\n");
        }
        fclose(img);
    }

    for (int i = 0; i < HEIGHT; i++) free(image[i]);
    free(image);
}

void generate_julia(const char *schedule_type, omp_sched_t omp_schedule, int chunk_size, int num_threads) {
    int **image = malloc(sizeof(int *) * HEIGHT);
    for (int i = 0; i < HEIGHT; i++) {
        image[i] = malloc(sizeof(int) * WIDTH);
    }

    double cx = -0.7, cy = 0.27015;
    omp_set_num_threads(num_threads);
    omp_set_schedule(omp_schedule, chunk_size);

    double total_start = omp_get_wtime();

    char logname[128];
    snprintf(logname, sizeof(logname), "logs/log_%s.txt", schedule_type);
    FILE *log = fopen(logname, "w");
    if (log) {
        fprintf(log, "Execution time per %d-row chunk (schedule: %s)\n", CHUNK_HEIGHT, schedule_type);
    }

    for (int start = 0; start < HEIGHT; start += CHUNK_HEIGHT) {
        int end = (start + CHUNK_HEIGHT > HEIGHT) ? HEIGHT : start + CHUNK_HEIGHT;
        double chunk_start = omp_get_wtime();

        #pragma omp parallel for schedule(runtime)
        for (int j = start; j < end; j++) {
            for (int i = 0; i < WIDTH; i++) {
                double x = 1.5 * (i - WIDTH / 2) / (0.5 * WIDTH);
                double y = (j - HEIGHT / 2) / (0.5 * HEIGHT);
                image[j][i] = julia(x, y, cx, cy);
            }
        }

        double chunk_end = omp_get_wtime();
        if (log) {
            fprintf(log, "Rows %4d-%4d: %.6f seconds\n", start, end - 1, chunk_end - chunk_start);
        }
    }

    if (log) fclose(log);

    double total_end = omp_get_wtime();
    double elapsed = total_end - total_start;

    char summary_name[128];
    snprintf(summary_name, sizeof(summary_name), "summaries/summary_%s.txt", schedule_type);
    FILE *info = fopen(summary_name, "w");
    if (info) {
        fprintf(info, "Schedule: %s\n", schedule_type);
        fprintf(info, "Execution time: %.6f seconds\n", elapsed);
        fprintf(info, "Threads used: %d\n", num_threads);
        fprintf(info, "Processors available: %d\n", omp_get_num_procs());
        fclose(info);
    }

    printf("Schedule %s finished in %.6f seconds using %d threads.\n", schedule_type, elapsed, num_threads);

    for (int i = 0; i < HEIGHT; i++) free(image[i]);
    free(image);
}

int main() {
    int num_threads = 8;

    generate_julia("static", omp_sched_static, 0, num_threads);
    generate_julia("dynamic", omp_sched_dynamic, 10, num_threads);
    generate_julia("guided", omp_sched_guided, 0, num_threads);

    //generate_reference_image("images/julia_reference.ppm");

    return 0;
}
