#ifndef WEAVETAB_GATE_H
#define WEAVETAB_GATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WtGate WtGate;

WtGate* wt_gate_new(const char* cache_dir, size_t threads);
void wt_gate_free(WtGate* gate);

/* Returns JSON string: {"choice":...,"confidence":...,"scores":[...]} or {"error":...}. Free with wt_gate_string_free. */
char* wt_gate_decide_text(
    const WtGate* gate,
    const char* text,
    const char* const* choices,
    size_t choices_len
);

char* wt_gate_decide_image(
    const WtGate* gate,
    const uint8_t* image_bytes,
    size_t image_len,
    const char* const* choices,
    size_t choices_len
);

void wt_gate_string_free(char* ptr);

#ifdef __cplusplus
}
#endif

#endif
