#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

int main() {
    const int N = 128, L = 512, S = 328;

    // 1. FP8 A (uint8), N×L
    {
        uint8_t *data = new uint8_t[N * L];
        for (int i = 0; i < N * L; i++) data[i] = rand() % 256;
        FILE *f = fopen("input/matrix_a_8.bin", "wb");
        fwrite(data, 1, N * L, f);
        fclose(f);
        delete[] data;
    }

    // 2. FP8 B_T (uint8), S×L (B转置后存储)
    {
        uint8_t *data = new uint8_t[S * L];
        for (int i = 0; i < S * L; i++) data[i] = rand() % 256;
        FILE *f = fopen("input/matrix_b_8.bin", "wb");
        fwrite(data, 1, S * L, f);
        fclose(f);
        delete[] data;
    }

    // 3. Lookup table (float), 256×256
    {
        float *data = new float[256 * 256];
        for (int i = 0; i < 256 * 256; i++) data[i] = (float)(rand() % 1000) / 100.0f;
        FILE *f = fopen("input/look_up_table_fp32.bin", "wb");
        fwrite(data, sizeof(float), 256 * 256, f);
        fclose(f);
        delete[] data;
    }

    // 4. FP16 A (__fp16), N×L
    {
        // store as binary16 (uint16)
        uint16_t *data = new uint16_t[N * L];
        for (int i = 0; i < N * L; i++) {
            float v = (float)(rand() % 1000) / 100.0f;
            // convert float to fp16 manually
            uint32_t bits;
            memcpy(&bits, &v, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            uint16_t mant = (bits >> 13) & 0x03ff;
            if (exp <= 0) { exp = 0; mant = 0; }
            else if (exp > 31) { exp = 31; mant = 0; }
            data[i] = sign | (exp << 10) | mant;
        }
        FILE *f = fopen("input/matrix_a_16.bin", "wb");
        fwrite(data, sizeof(uint16_t), N * L, f);
        fclose(f);
        delete[] data;
    }

    // 5. FP16 B_T (__fp16), S×L
    {
        uint16_t *data = new uint16_t[S * L];
        for (int i = 0; i < S * L; i++) {
            float v = (float)(rand() % 1000) / 100.0f;
            uint32_t bits;
            memcpy(&bits, &v, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            uint16_t mant = (bits >> 13) & 0x03ff;
            if (exp <= 0) { exp = 0; mant = 0; }
            else if (exp > 31) { exp = 31; mant = 0; }
            data[i] = sign | (exp << 10) | mant;
        }
        FILE *f = fopen("input/matrix_b_16.bin", "wb");
        fwrite(data, sizeof(uint16_t), S * L, f);
        fclose(f);
        delete[] data;
    }

    // 6. INT8 A (int8), N×L
    {
        int8_t *data = new int8_t[N * L];
        for (int i = 0; i < N * L; i++) data[i] = (int8_t)(rand() % 256 - 128);
        FILE *f = fopen("input/matrix_a_i8.bin", "wb");
        fwrite(data, 1, N * L, f);
        fclose(f);
        delete[] data;
    }

    // 7. INT8 B_T (int8), S×L
    {
        int8_t *data = new int8_t[S * L];
        for (int i = 0; i < S * L; i++) data[i] = (int8_t)(rand() % 256 - 128);
        FILE *f = fopen("input/matrix_b_i8.bin", "wb");
        fwrite(data, 1, S * L, f);
        fclose(f);
        delete[] data;
    }

    printf("All input files generated.\n");
    return 0;
}
