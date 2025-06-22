#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

#define WIDTH 19200
#define HEIGHT 19200
#define MAX_ITER 10000
#define CHUNK_HEIGHT 1200

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

void generate_julia(const char *schedule_type, omp_sched_t omp_schedule, int chunk_size,
                    int num_threads, int rank, int start_row, int end_row) {

    int rows_to_process = end_row - start_row;
    int **image = malloc(sizeof(int *) * rows_to_process);
    for (int i = 0; i < rows_to_process; i++) {
        image[i] = malloc(sizeof(int) * WIDTH);
    }

    double cx = -0.7, cy = 0.27015;

    omp_set_num_threads(num_threads);
    omp_set_schedule(omp_schedule, chunk_size);

    double total_start = omp_get_wtime();

    char logname[128];
    snprintf(logname, sizeof(logname), "logs/log_%s_rank%d.txt", schedule_type, rank);
    FILE *log = fopen(logname, "w");
    if (log) {
        fprintf(log, "Execution time per %d-row chunk (schedule: %s) - Rank %d\n",
                CHUNK_HEIGHT, schedule_type, rank);
    }

    for (int start = 0; start < rows_to_process; start += CHUNK_HEIGHT) {
        int local_start = start;
        int local_end = (start + CHUNK_HEIGHT > rows_to_process) ? rows_to_process : start + CHUNK_HEIGHT;

        double chunk_start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp for schedule(runtime)
            for (int j = local_start; j < local_end; j++) {
                for (int i = 0; i < WIDTH; i++) {
                    double x = 1.5 * (i - WIDTH / 2) / (0.5 * WIDTH);
                    double y = (j + start_row - HEIGHT / 2) / (0.5 * HEIGHT);
                    image[j][i] = julia(x, y, cx, cy);
                }
            }
        }

        double chunk_end = omp_get_wtime();
        if (log) {
            fprintf(log, "Rank %d, Rows %4d-%4d: %.6f seconds\n",
                    rank, start_row + local_start, start_row + local_end - 1, chunk_end - chunk_start);
        }
    }

    if (log) fclose(log);

    double total_end = omp_get_wtime();
    double elapsed = total_end - total_start;

    char summary_name[128];
    snprintf(summary_name, sizeof(summary_name), "summaries/summary_%s_rank%d.txt", schedule_type, rank);
    FILE *summary = fopen(summary_name, "w");
    if (summary) {
        fprintf(summary, "Schedule: %s (rank %d)\n", schedule_type, rank);
        fprintf(summary, "Execution time: %.6f seconds\n", elapsed);
        fprintf(summary, "Threads used: %d\n", num_threads);
        fprintf(summary, "Processors available: %d\n", omp_get_num_procs());
        fclose(summary);
    }

    printf("Rank %d: Schedule %s finished in %.6f seconds using %d threads.\n",
           rank, schedule_type, elapsed, num_threads);

    for (int i = 0; i < rows_to_process; i++) free(image[i]);
    free(image);
}

void generate_reference_image() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Generating reference image with chunk lines...\n");

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

        FILE *img = fopen("images/reference_with_lines.ppm", "w");
        if (!img) {
            perror("Failed to open reference image file");
            for (int i = 0; i < HEIGHT; i++) free(image[i]);
            free(image);
            return;
        }

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
        printf("Reference image saved.\n");

        for (int i = 0; i < HEIGHT; i++) free(image[i]);
        free(image);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_chunks = HEIGHT / CHUNK_HEIGHT;

    if (size != total_chunks) {
        if (rank == 0) {
            fprintf(stderr, "Please run the program with exactly %d MPI processes (matching chunk count).\n", total_chunks);
        }
        MPI_Finalize();
        return 1;
    }

    int start_row = rank * CHUNK_HEIGHT;
    int end_row = (rank == total_chunks - 1) ? HEIGHT : (rank + 1) * CHUNK_HEIGHT;
    int num_threads = 2;

    generate_julia("static", omp_sched_static, 0, num_threads, rank, start_row, end_row);
    generate_julia("dynamic", omp_sched_dynamic, 10, num_threads, rank, start_row, end_row);
    generate_julia("guided", omp_sched_guided, 0, num_threads, rank, start_row, end_row);

    //generate_reference_image();

    MPI_Finalize();
    return 0;
}
