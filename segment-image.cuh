#ifndef SEGMENT_IMAGE_CUH
#define SEGMENT_IMAGE_CUH

// Librerías necesarias
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include "image.h"
#include "misc.h"
#include "filter.h"
#include "segment-graph.h"

// Macro para verificar errores de CUDA
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

/*
 * Calcula la diferencia de color entre dos píxeles (usado para generar pesos de las aristas).
 */
__device__ static inline float diff_device(float *r, float *g, float *b,
                                          int x1, int y1, int x2, int y2, int width) {
    return sqrtf(square(r[y1 * width + x1] - r[y2 * width + x2]) +
                 square(g[y1 * width + x1] - g[y2 * width + x2]) +
                 square(b[y1 * width + x1] - b[y2 * width + x2]));
}

/*
 * Kernel CUDA que genera las aristas entre píxeles vecinos (derecha y abajo),
 * asignando el peso según la diferencia de color.
 */
__global__ void generate_edges_kernel(float *d_smooth_r, float *d_smooth_g, float *d_smooth_b,
                                      edge *d_edges, int width, int height, int *d_num_edges) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // Arista hacia la derecha
        if (x < width - 1) {
            float w = diff_device(d_smooth_r, d_smooth_g, d_smooth_b, x, y, x + 1, y, width);
            int i = atomicAdd(d_num_edges, 1);
            d_edges[i].a = idx;
            d_edges[i].b = idx + 1;
            d_edges[i].w = w;
        }

        // Arista hacia abajo
        if (y < height - 1) {
            float w = diff_device(d_smooth_r, d_smooth_g, d_smooth_b, x, y, x, y + 1, width);
            int i = atomicAdd(d_num_edges, 1);
            d_edges[i].a = idx;
            d_edges[i].b = idx + width;
            d_edges[i].w = w;
        }
    }
}

/*
 * Kernel CUDA para escribir los colores finales de cada componente en la imagen de salida.
 */
__global__ void write_output_kernel(rgb *d_output, int *d_comp_map, rgb *d_colors, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;
    int idx = y * width + x;
    int comp = d_comp_map[idx];
    d_output[idx] = d_colors[comp];
}

/*
 * Función principal que realiza la segmentación de una imagen.
 * 
 * Parámetros:
 *   - im: imagen de entrada.
 *   - sigma: parámetro de suavizado para el filtro Gaussiano.
 *   - c: constante del umbral para la segmentación.
 *   - min_size: tamaño mínimo de cada componente.
 *   - num_ccs: número de componentes conectados resultantes (output).
 * 
 * Retorna:
 *   - Imagen segmentada, con un color aleatorio asignado a cada componente.
 */
image<rgb> *segment_image(image<rgb> *im, float sigma, float c, int min_size, int *num_ccs) {
    int width = im->width(), height = im->height();
    int num_pixels = width * height;

    // Separar canales RGB
    image<float> *r_tmp = new image<float>(width, height);
    image<float> *g_tmp = new image<float>(width, height);
    image<float> *b_tmp = new image<float>(width, height);

    for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
        rgb pix = imRef(im, x, y);
        imRef(r_tmp, x, y) = pix.r;
        imRef(g_tmp, x, y) = pix.g;
        imRef(b_tmp, x, y) = pix.b;
    }

    // Aplicar filtro Gaussiano (suavizado)
    image<float> *smooth_r = smooth(r_tmp, sigma);
    image<float> *smooth_g = smooth(g_tmp, sigma);
    image<float> *smooth_b = smooth(b_tmp, sigma);
    delete r_tmp; delete g_tmp; delete b_tmp;

    // Copiar datos suavizados a memoria de GPU
    float *d_r, *d_g, *d_b;
    cudaMalloc(&d_r, num_pixels * sizeof(float));
    cudaMalloc(&d_g, num_pixels * sizeof(float));
    cudaMalloc(&d_b, num_pixels * sizeof(float));
    cudaMemcpy(d_r, smooth_r->data, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, smooth_g->data, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, smooth_b->data, num_pixels * sizeof(float), cudaMemcpyHostToDevice);
    delete smooth_r; delete smooth_g; delete smooth_b;

    // Preparar estructura para aristas
    int max_edges = 2 * num_pixels;
    edge *d_edges;
    int *d_num_edges;
    cudaMalloc(&d_edges, max_edges * sizeof(edge));
    cudaMalloc(&d_num_edges, sizeof(int));
    cudaMemset(d_num_edges, 0, sizeof(int));

    // Ejecutar kernel que genera las aristas entre píxeles
    dim3 block(32, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    generate_edges_kernel<<<grid, block>>>(d_r, d_g, d_b, d_edges, width, height, d_num_edges);
    cudaDeviceSynchronize(); cudaCheckError();

    // Copiar aristas generadas a CPU
    int num_edges;
    cudaMemcpy(&num_edges, d_num_edges, sizeof(int), cudaMemcpyDeviceToHost);
    edge *edges_host = new edge[num_edges];
    cudaMemcpy(edges_host, d_edges, num_edges * sizeof(edge), cudaMemcpyDeviceToHost);
    cudaFree(d_edges); cudaFree(d_num_edges); cudaFree(d_r); cudaFree(d_g); cudaFree(d_b);

    // Ejecutar segmentación en CPU
    universe *u = segment_graph(num_pixels, num_edges, edges_host, c);
    for (int i = 0; i < num_edges; ++i) {
        int a = u->find(edges_host[i].a);
        int b = u->find(edges_host[i].b);
        if (a != b && (u->size(a) < min_size || u->size(b) < min_size)) {
            u->join(a, b);
        }
    }
    delete[] edges_host;
    *num_ccs = u->num_sets();  // Guardar número de componentes

    // Crear mapa de componentes
    int *components_map_host = new int[num_pixels];
    for (int i = 0; i < num_pixels; ++i)
        components_map_host[i] = u->find(i);

    // Asignar color aleatorio a cada componente
    std::unordered_map<int, rgb> component_colors;
    for (int i = 0; i < num_pixels; ++i) {
        int comp = components_map_host[i];
        if (component_colors.find(comp) == component_colors.end()) {
            rgb c = { static_cast<uchar>(rand() % 256),
                      static_cast<uchar>(rand() % 256),
                      static_cast<uchar>(rand() % 256) };
            component_colors[comp] = c;
        }
    }

    // Crear array de colores por píxel
    rgb *colors_host = new rgb[num_pixels];
    for (int i = 0; i < num_pixels; ++i)
        colors_host[i] = component_colors[components_map_host[i]];

    // Copiar datos a GPU para escribir imagen final
    int *d_comp_map;
    rgb *d_colors, *d_output;
    cudaMalloc(&d_comp_map, num_pixels * sizeof(int));
    cudaMalloc(&d_colors, num_pixels * sizeof(rgb));
    cudaMalloc(&d_output, num_pixels * sizeof(rgb));
    cudaMemcpy(d_comp_map, components_map_host, num_pixels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, colors_host, num_pixels * sizeof(rgb), cudaMemcpyHostToDevice);
    delete[] components_map_host;
    delete[] colors_host;

    // Ejecutar kernel que escribe los colores por píxel
    write_output_kernel<<<grid, block>>>(d_output, d_comp_map, d_colors, width, height);
    cudaDeviceSynchronize(); cudaCheckError();

    // Copiar imagen de salida a CPU
    image<rgb> *output = new image<rgb>(width, height);
    cudaMemcpy(output->data, d_output, num_pixels * sizeof(rgb), cudaMemcpyDeviceToHost);
    cudaFree(d_comp_map); cudaFree(d_colors); cudaFree(d_output);
    delete u;

    return output;
}

#endif

