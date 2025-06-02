#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.cuh"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp> // Incluye OpenCV para visualización

// Programa principal que realiza la segmentación de una imagen utilizando CUDA.
int main(int argc, char **argv) {
    // Verifica que se hayan pasado los argumentos necesarios al programa.
    if (argc != 6) {
        fprintf(stderr, "usage: %s sigma k min input(ppm) output(ppm)\n", argv[0]);
        return 1;
    }

    // Convierte los argumentos de entrada a los valores correspondientes.
    float sigma = atof(argv[1]); // Parámetro sigma para suavizado.
    float k = atof(argv[2]);     // Parámetro k para el algoritmo de segmentación.
    int min_size = atoi(argv[3]); // Tamaño mínimo de los componentes segmentados.

    // Carga la imagen de entrada desde un archivo PPM.
    printf("loading input image.\n");
    image<rgb> *input = loadPPM(argv[4]);

    // Convierte la imagen de entrada a formato OpenCV para visualización.
    cv::Mat input_mat(input->height(), input->width(), CV_8UC3, input->data);
    cv::cvtColor(input_mat, input_mat, cv::COLOR_RGB2BGR); // Convierte de RGB a BGR para OpenCV.

    // Muestra la imagen original.
    cv::imshow("Original Image", input_mat);

    // Inicia el proceso de segmentación de la imagen.
    printf("processing\n");
    auto start = std::chrono::high_resolution_clock::now(); // Marca el inicio del tiempo de ejecución.

    int num_ccs; // Variable para almacenar el número de componentes conectados.
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); // Realiza la segmentación de la imagen.

    auto end = std::chrono::high_resolution_clock::now(); // Marca el final del tiempo de ejecución.
    std::chrono::duration<double> elapsed = end - start; // Calcula el tiempo transcurrido.

    // Imprime el tiempo de ejecución del algoritmo CUDA.
    printf("Tiempo de ejecución CUDA: %.6f segundos\n", elapsed.count());

    // Calcula la memoria utilizada por las imágenes.
    size_t input_memory = input->width() * input->height() * sizeof(rgb);
    size_t seg_memory = seg->width() * seg->height() * sizeof(rgb);
    printf("Memoria utilizada por la imagen de entrada: %zu bytes\n", input_memory);
    printf("Memoria utilizada por la imagen segmentada: %zu bytes\n", seg_memory);

    // Convierte la imagen segmentada a formato OpenCV para visualización.
    cv::Mat seg_mat(seg->height(), seg->width(), CV_8UC3, seg->data);
    cv::cvtColor(seg_mat, seg_mat, cv::COLOR_RGB2BGR); // Convierte de RGB a BGR para OpenCV.

    // Muestra la imagen segmentada.
    cv::imshow("Segmented Image", seg_mat);

    // Guarda la imagen segmentada en un archivo PPM.
    savePPM(seg, argv[5]);

    // Imprime el número de componentes conectados encontrados.
    printf("Los %d componentes\n", num_ccs);
    printf("HECHO!.\n");

    // Espera a que el usuario cierre las ventanas de visualización.
    cv::waitKey(0);

    // Libera la memoria utilizada por las imágenes.
    delete input;
    delete seg;

    return 0; // Finaliza el programa.
}
