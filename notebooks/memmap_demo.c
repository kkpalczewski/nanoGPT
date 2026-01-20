/**
 * memmap_demo.c - Understanding memory-mapped files at the OS level
 * 
 * This is what numpy.memmap does under the hood using the mmap() system call.
 * 
 * Compile: gcc -o memmap_demo memmap_demo.c
 * Run:     ./memmap_demo ../data/shakespeare/train.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

// Token type (matches numpy uint16)
typedef uint16_t token_t;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <path_to_bin_file>\n", argv[0]);
        printf("Example: %s ../data/shakespeare/train.bin\n", argv[0]);
        return 1;
    }
    
    const char *filepath = argv[1];
    
    printf("=== Memory-Mapped Files in C ===\n\n");
    
    // -------------------------------------------------------------------------
    // Step 1: Open the file
    // -------------------------------------------------------------------------
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        return 1;
    }
    
    size_t file_size = sb.st_size;
    size_t n_tokens = file_size / sizeof(token_t);
    
    printf("File: %s\n", filepath);
    printf("  Size: %zu bytes (%.2f MB)\n", file_size, file_size / 1024.0 / 1024.0);
    printf("  Tokens: %zu\n\n", n_tokens);
    
    // -------------------------------------------------------------------------
    // Step 2: Memory-map the file (THIS IS THE KEY PART)
    // -------------------------------------------------------------------------
    printf("--- Calling mmap() ---\n");
    printf("This maps the file to virtual memory WITHOUT loading it into RAM.\n\n");
    
    /*
     * mmap() arguments:
     *   addr   = NULL     (let OS choose the address)
     *   length = file_size (map entire file)
     *   prot   = PROT_READ (read-only access)
     *   flags  = MAP_PRIVATE (changes won't write to file)
     *   fd     = file descriptor
     *   offset = 0 (start from beginning)
     * 
     * Returns: pointer to mapped memory region
     */
    token_t *data = (token_t *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    if (data == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        return 1;
    }
    
    // Can close fd after mmap - the mapping persists
    close(fd);
    
    printf("mmap() returned address: %p\n", (void *)data);
    printf("At this point: file is NOT in RAM, just mapped to virtual addresses.\n\n");
    
    // -------------------------------------------------------------------------
    // Step 3: Access data (triggers page faults -> OS loads from disk)
    // -------------------------------------------------------------------------
    printf("--- Accessing data (triggers page loading) ---\n\n");
    
    printf("First 10 tokens:\n  ");
    for (int i = 0; i < 10; i++) {
        printf("%u ", data[i]);  // Page fault here! OS loads the page containing data[i]
    }
    printf("\n\n");
    
    // Random access - like training
    printf("Random access (simulating training batches):\n");
    srand(time(NULL));
    for (int batch = 0; batch < 3; batch++) {
        size_t idx = rand() % (n_tokens - 1024);
        printf("  data[%zu:%zu] = [%u, %u, %u, ...]\n", 
               idx, idx + 1024, data[idx], data[idx + 1], data[idx + 2]);
    }
    printf("\n");
    
    // -------------------------------------------------------------------------
    // Step 4: Stats about virtual memory pages
    // -------------------------------------------------------------------------
    printf("--- Virtual Memory Pages ---\n\n");
    
    long page_size = sysconf(_SC_PAGESIZE);
    size_t n_pages = (file_size + page_size - 1) / page_size;
    
    printf("System page size: %ld bytes (%ld KB)\n", page_size, page_size / 1024);
    printf("File spans: %zu pages\n", n_pages);
    printf("Tokens per page: %ld\n\n", page_size / sizeof(token_t));
    
    printf("How it works:\n");
    printf("  1. mmap() creates virtual address mappings (no RAM used)\n");
    printf("  2. First access to data[i] -> PAGE FAULT\n");
    printf("  3. OS loads the 4KB page containing data[i] from disk\n");
    printf("  4. Subsequent accesses to same page are instant (in RAM)\n");
    printf("  5. OS can evict unused pages when RAM is needed\n\n");
    
    // -------------------------------------------------------------------------
    // Step 5: Compare with traditional read()
    // -------------------------------------------------------------------------
    printf("--- Comparison: mmap() vs read() ---\n\n");
    
    printf("Traditional read():\n");
    printf("  - Loads entire file into RAM immediately\n");
    printf("  - RAM usage = file size\n");
    printf("  - Good for: small files, sequential access\n\n");
    
    printf("Memory-mapped mmap():\n");
    printf("  - Maps file to virtual addresses (no immediate RAM)\n");
    printf("  - RAM usage = only accessed pages\n");
    printf("  - Good for: large files, random access, shared memory\n");
    printf("  - This is what numpy.memmap() uses!\n\n");
    
    // -------------------------------------------------------------------------
    // Step 6: Show byte layout (like numpy uint16)
    // -------------------------------------------------------------------------
    printf("--- Byte Layout (uint16 little-endian) ---\n\n");
    
    unsigned char *raw = (unsigned char *)data;
    printf("First 20 bytes (hex): ");
    for (int i = 0; i < 20; i++) {
        printf("%02x ", raw[i]);
    }
    printf("\n\n");
    
    printf("Interpreted as uint16:\n");
    for (int i = 0; i < 10; i++) {
        uint16_t val = raw[i*2] | (raw[i*2 + 1] << 8);  // little-endian
        printf("  bytes[%d:%d] = %02x %02x -> %u\n", 
               i*2, i*2+2, raw[i*2], raw[i*2+1], val);
    }
    printf("\n");
    
    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    if (munmap(data, file_size) == -1) {
        perror("Error unmapping");
    }
    
    printf("Done! File unmapped.\n");
    return 0;
}
