#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 19800
#define HEIGHT 19800
#define MAX_ITER 10000
#define CHUNK_HEIGHT 4950

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

void generate_julia_chunk(const char *tag, int rank, int start_row, int end_row) {
    int rows_to_process = end_row - start_row;
    int **image = malloc(sizeof(int *) * rows_to_process);
    for (int i = 0; i < rows_to_process; i++) {
        image[i] = malloc(sizeof(int) * WIDTH);
    }

    double cx = -0.7, cy = 0.27015;
    double chunk_start = MPI_Wtime();

    for (int j = 0; j < rows_to_process; j++) {
        for (int i = 0; i < WIDTH; i++) {
            double x = 1.5 * (i - WIDTH / 2) / (0.5 * WIDTH);
            double y = (j + start_row - HEIGHT / 2) / (0.5 * HEIGHT);
            image[j][i] = julia(x, y, cx, cy);
        }
    }

    double chunk_end = MPI_Wtime();

    char logname[128];
    snprintf(logname, sizeof(logname), "logs/log_%s_rank%d.txt", tag, rank);
    FILE *log = fopen(logname, "w");
    if (log) {
        fprintf(log, "Rank %d processed rows %d-%d in %.6f seconds\n", rank, start_row, end_row - 1, chunk_end - chunk_start);
        fclose(log);
    }

    char summary_name[128];
    snprintf(summary_name, sizeof(summary_name), "summaries/summary_%s_rank%d.txt", tag, rank);
    FILE *summary = fopen(summary_name, "w");
    if (summary) {
        fprintf(summary, "Tag: %s (rank %d)\n", tag, rank);
        fprintf(summary, "Processed rows: %d to %d\n", start_row, end_row - 1);
        fprintf(summary, "Execution time: %.6f seconds\n", chunk_end - chunk_start);
        fclose(summary);
    }

    for (int i = 0; i < rows_to_process; i++) free(image[i]);
    free(image);

    printf("Rank %d: finished rows %d-%d in %.6f seconds.\n", rank, start_row, end_row - 1, chunk_end - chunk_start);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int total_chunks = (HEIGHT + CHUNK_HEIGHT - 1) / CHUNK_HEIGHT;

    if (size != total_chunks) {
        if (rank == 0) {
            fprintf(stderr, "Error: Run the program with exactly %d MPI processes (one per chunk of %d rows).\n", total_chunks, CHUNK_HEIGHT);
        }
        MPI_Finalize();
        return 1;
    }

    int start_row = rank * CHUNK_HEIGHT;
    int end_row = (rank == total_chunks - 1) ? HEIGHT : (rank + 1) * CHUNK_HEIGHT;

    generate_julia_chunk("mpi", rank, start_row, end_row);

    //generate_reference_image("images/julia_reference.ppm");

    MPI_Finalize();
    return 0;
}
