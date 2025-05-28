#include <SDL.h>
#include <SDL_ttf.h>
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_PLOT_WIDTH 400
#define DEFAULT_PLOT_HEIGHT 300
#define SAMPLE_RATE 44100
#define NUM_SAMPLES 1024
#define FONT_SIZE 14
#define MAX_SIGNAL_TYPES 5
#define GRID_COLOR 40
#define AXIS_COLOR 100
#define TEXT_COLOR 200
#define PLOT_BG_COLOR 30
#define AUDIO_BUFFER_SIZE 4096
#define DEFAULT_WINDOW_PADDING 10
#define NUM_MFCC 13
#define NUM_MEL_FILTERS 20
#define MIN_FREQ 20
#define MAX_FREQ 8000
#define MFCC_HISTORY 30

typedef enum {
    SIGNAL_SINE,
    SIGNAL_SQUARE,
    SIGNAL_SAWTOOTH,
    SIGNAL_TRIANGLE,
    SIGNAL_CUSTOM
} SignalType;

typedef struct {
    double frequency;
    double amplitude;
    double phase;
    int enabled;
} SignalComponent;

typedef struct {
    SignalType type;
    SignalComponent components[4];
    double noise_level;
    double decay_rate;
    int num_components;
} SignalConfig;

typedef struct {
    SDL_Color magnitude;
    SDL_Color phase;
    SDL_Color real;
    SDL_Color imaginary;
    SDL_Color time;
} PlotColors;

typedef struct {
    SDL_Renderer* renderer;
    SDL_Rect rect;
    const char* title;
} PlotArea;

typedef struct {
    float* buffer;
    int write_pos;
    int read_pos;
    int size;
    SDL_mutex* mutex;
} AudioBuffer;

typedef struct {
    double* mel_filters;
    int num_filters;
    int num_coeffs;
    double* history;
    int history_pos;
} MFCCConfig;

TTF_Font* font = NULL;
SignalConfig current_config;
int is_running = 1;
int show_phase = 1;
int show_magnitude = 1;
int show_real = 1;
int show_imaginary = 1;
int is_recording = 0;

double time_domain_scale = 1.0;

PlotColors colors = {
    {255, 255, 255, 255},  // magnitude
    {255, 100, 100, 255},  // phase
    {100, 255, 100, 255},  // real
    {100, 100, 255, 255},  // imaginary
    {255, 255, 255, 255}   // time
};

SDL_Window* main_window = NULL;
SDL_Renderer* main_renderer = NULL;
PlotArea plots[6] = {0};  // Time, Magnitude, Phase, Real, Imaginary, MFCC
AudioBuffer audio_buffer = {0};

int PLOT_WIDTH = DEFAULT_PLOT_WIDTH;
int PLOT_HEIGHT = DEFAULT_PLOT_HEIGHT;
int WINDOW_PADDING = DEFAULT_WINDOW_PADDING;

MFCCConfig mfcc_config = {0};

void init_audio_buffer() {
    audio_buffer.buffer = (float*)malloc(AUDIO_BUFFER_SIZE * sizeof(float));
    audio_buffer.size = AUDIO_BUFFER_SIZE;
    audio_buffer.write_pos = 0;
    audio_buffer.read_pos = 0;
    audio_buffer.mutex = SDL_CreateMutex();
    memset(audio_buffer.buffer, 0, AUDIO_BUFFER_SIZE * sizeof(float));
}

void cleanup_audio_buffer() {
    if (audio_buffer.buffer) {
        free(audio_buffer.buffer);
    }
    if (audio_buffer.mutex) {
        SDL_DestroyMutex(audio_buffer.mutex);
    }
}

void audio_callback(void* userdata, Uint8* stream, int len) {
    if (!is_recording) return;
    
    float* audio_data = (float*)stream;
    int num_samples = len / sizeof(float);
    
    SDL_LockMutex(audio_buffer.mutex);
    for (int i = 0; i < num_samples; i++) {
        audio_buffer.buffer[audio_buffer.write_pos] = audio_data[i];
        audio_buffer.write_pos = (audio_buffer.write_pos + 1) % audio_buffer.size;
    }
    SDL_UnlockMutex(audio_buffer.mutex);
}

void init_signal_config() {
    current_config.type = SIGNAL_SINE;
    current_config.noise_level = 0.1;
    current_config.decay_rate = 2.0;
    current_config.num_components = 4;
    
    current_config.components[0] = (SignalComponent){5.0, 2.0, 0.0, 1};
    current_config.components[1] = (SignalComponent){10.0, 1.5, 0.0, 1};
    current_config.components[2] = (SignalComponent){20.0, 0.8, 0.0, 1};
    current_config.components[3] = (SignalComponent){30.0, 0.4, 0.0, 1};
}

double complex* compute_dft(double* input, int n) {
    double complex* output = (double complex*)malloc(n * sizeof(double complex));
    if (!output) return NULL;

    for (int k = 0; k < n; k++) {
        output[k] = 0;
        for (int j = 0; j < n; j++) {
            double angle = 2 * M_PI * k * j / n;
            output[k] += input[j] * (cos(angle) - I * sin(angle));
        }
    }
    return output;
}

void get_audio_data(double* output, int n) {
    SDL_LockMutex(audio_buffer.mutex);
    int start = (audio_buffer.write_pos - n + audio_buffer.size) % audio_buffer.size;
    for (int i = 0; i < n; i++) {
        int idx = (start + i) % audio_buffer.size;
        output[i] = audio_buffer.buffer[idx];
    }
    SDL_UnlockMutex(audio_buffer.mutex);
}

void render_text(SDL_Renderer* renderer, const char* text, int x, int y, SDL_Color color) {
    SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
    if (!surface) return;
    
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        SDL_FreeSurface(surface);
        return;
    }
    
    SDL_Rect rect = {x, y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &rect);
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

void draw_grid(SDL_Renderer* renderer, SDL_Rect* rect, int window_width, int window_height, int x_divs, int y_divs) {
    SDL_SetRenderDrawColor(renderer, GRID_COLOR, GRID_COLOR, GRID_COLOR, 255);
    for (int i = 1; i < x_divs; i++) {
        int x = rect->x + i * window_width / x_divs;
        SDL_RenderDrawLine(renderer, x, rect->y, x, rect->y + window_height);
    }
    for (int i = 1; i < y_divs; i++) {
        int y = rect->y + i * window_height / y_divs;
        SDL_RenderDrawLine(renderer, rect->x, y, rect->x + window_width, y);
    }
    SDL_SetRenderDrawColor(renderer, AXIS_COLOR, AXIS_COLOR, AXIS_COLOR, 255);
    SDL_RenderDrawLine(renderer, rect->x, rect->y + window_height/2, rect->x + window_width, rect->y + window_height/2);
    SDL_RenderDrawLine(renderer, rect->x + window_width/2, rect->y, rect->x + window_width/2, rect->y + window_height);
}

void draw_ticks(SDL_Renderer* renderer, SDL_Rect* rect, int window_width, int window_height, int x_divs, int y_divs) {
    SDL_SetRenderDrawColor(renderer, TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, 255);
    for (int i = 0; i <= x_divs; i++) {
        int x = rect->x + i * window_width / x_divs;
        SDL_RenderDrawLine(renderer, x, rect->y + window_height/2 - 2, x, rect->y + window_height/2 + 2);
    }
    for (int i = 0; i <= y_divs; i++) {
        int y = rect->y + i * window_height / y_divs;
        SDL_RenderDrawLine(renderer, rect->x + window_width/2 - 2, y, rect->x + window_width/2 + 2, y);
    }
}

void draw_axis_numbers(SDL_Renderer* renderer, SDL_Rect* rect, int window_width, int window_height, int x_divs, int y_divs, double x_max, double y_max, int is_freq_domain) {
    char text[32];
    SDL_Color color = {TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, 255};
    for (int i = 0; i <= x_divs; i++) {
        double value = is_freq_domain ? i * x_max / x_divs : (i - x_divs/2) * x_max / (x_divs/2);
        sprintf(text, "%.1f", value);
        int x = rect->x + i * window_width / x_divs;
        render_text(renderer, text, x - 10, rect->y + window_height/2 + 3, color);
    }
    for (int i = 0; i <= y_divs; i++) {
        double value = (y_divs/2 - i) * y_max / (y_divs/2);
        sprintf(text, "%.1f", value);
        int y = rect->y + i * window_height / y_divs;
        render_text(renderer, text, rect->x + window_width/2 + 3, y - 8, color);
    }
}

void plot_function(SDL_Renderer* renderer, SDL_Rect* rect, double* data, int n, int window_width, int window_height, SDL_Color color) {
    if (!data || n <= 0) return;
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    double max_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(data[i]) > max_val) max_val = fabs(data[i]);
    }
    if (max_val == 0.0) max_val = 1.0;
    for (int i = 0; i < n - 1; i++) {
        int x1 = rect->x + i * window_width / n;
        int y1 = rect->y + window_height/2 - (int)(data[i] * window_height/2 / max_val);
        int x2 = rect->x + (i + 1) * window_width / n;
        int y2 = rect->y + window_height/2 - (int)(data[i + 1] * window_height/2 / max_val);
        SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
    }
}

void draw_controls(SDL_Renderer* renderer, SDL_Rect* rect) {
    SDL_Color color = {TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, 255};
    char text[256];
    
    sprintf(text, "Controls: Space: Toggle Recording | P/M/R/I: Toggle Plots | Esc: Quit");
    render_text(renderer, text, rect->x + 5, rect->y + rect->h - 20, color);
    
    sprintf(text, "Status: %s", is_recording ? "Recording" : "Stopped");
    render_text(renderer, text, rect->x + 5, rect->y + rect->h - 35, color);
}

void handle_input(SDL_Event* event) {
    if (event->type == SDL_KEYDOWN) {
        switch (event->key.keysym.sym) {
            case SDLK_SPACE: is_recording = !is_recording; break;
            case SDLK_p: show_phase = !show_phase; break;
            case SDLK_m: show_magnitude = !show_magnitude; break;
            case SDLK_r: show_real = !show_real; break;
            case SDLK_i: show_imaginary = !show_imaginary; break;
            case SDLK_ESCAPE: is_running = 0; break;
        }
    }
}

void draw_plot_area(SDL_Renderer* renderer, SDL_Rect* rect) {
    SDL_SetRenderDrawColor(renderer, PLOT_BG_COLOR, PLOT_BG_COLOR, PLOT_BG_COLOR, 255);
    SDL_RenderFillRect(renderer, rect);
}

void init_mfcc() {
    mfcc_config.num_filters = NUM_MEL_FILTERS;
    mfcc_config.num_coeffs = NUM_MFCC;
    mfcc_config.mel_filters = (double*)malloc(NUM_MEL_FILTERS * (NUM_SAMPLES/2 + 1) * sizeof(double));
    mfcc_config.history = (double*)malloc(NUM_MFCC * MFCC_HISTORY * sizeof(double));
    mfcc_config.history_pos = 0;
    memset(mfcc_config.history, 0, NUM_MFCC * MFCC_HISTORY * sizeof(double));
    
    double min_mel = 2595 * log10(1 + MIN_FREQ/700.0);
    double max_mel = 2595 * log10(1 + MAX_FREQ/700.0);
    double mel_step = (max_mel - min_mel) / (NUM_MEL_FILTERS + 1);
    
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        double mel_center = min_mel + (i + 1) * mel_step;
        double freq_center = 700 * (pow(10, mel_center/2595) - 1);
        double freq_width = freq_center * 0.2;
        
        for (int j = 0; j < NUM_SAMPLES/2 + 1; j++) {
            double freq = j * SAMPLE_RATE / NUM_SAMPLES;
            double mel = 2595 * log10(1 + freq/700.0);
            double weight = 0;
            
            if (fabs(mel - mel_center) < mel_step) {
                weight = 1 - fabs(mel - mel_center) / mel_step;
            }
            
            mfcc_config.mel_filters[i * (NUM_SAMPLES/2 + 1) + j] = weight;
        }
    }
}

void compute_mfcc(double* magnitude, double* mfcc) {
    double* mel_energies = (double*)malloc(NUM_MEL_FILTERS * sizeof(double));
    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        mel_energies[i] = 0;
        for (int j = 0; j < NUM_SAMPLES/2 + 1; j++) {
            mel_energies[i] += magnitude[j] * mfcc_config.mel_filters[i * (NUM_SAMPLES/2 + 1) + j];
        }
        mel_energies[i] = log(mel_energies[i] + 1e-10);
    }
    for (int i = 0; i < NUM_MFCC; i++) {
        mfcc[i] = 0;
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            mfcc[i] += mel_energies[j] * cos(M_PI * i * (2 * j + 1) / (2 * NUM_MEL_FILTERS));
        }
        mfcc[i] *= 4.0;
    }
    memcpy(mfcc_config.history + mfcc_config.history_pos * NUM_MFCC, mfcc, NUM_MFCC * sizeof(double));
    mfcc_config.history_pos = (mfcc_config.history_pos + 1) % MFCC_HISTORY;
    free(mel_energies);
}

void get_color_for_value(double value, double max_val, SDL_Color* color) {
    value = fabs(value) / max_val;
    if (value < 0) value = 0;
    if (value > 1) value = 1;

    double r = 0, g = 0, b = 0;

    if (value < 0.125) {
        b = 0.5 + 4.0 * value; 
    } else if (value < 0.375) {
        b = 1.0;
        g = 4.0 * (value - 0.125); 
    } else if (value < 0.625) {
        g = 1.0;
        b = 1.0 - 4.0 * (value - 0.375);
        r = 4.0 * (value - 0.375);
    } else if (value < 0.875) {
        r = 1.0;
        g = 1.0 - 4.0 * (value - 0.625);
    } else {
        r = 1.0 - 4.0 * (value - 0.875);
    }

    color->r = (Uint8)(r * 255);
    color->g = (Uint8)(g * 255);
    color->b = (Uint8)(b * 255);
    color->a = 255;
}

void draw_mfcc_heatmap(SDL_Renderer* renderer, SDL_Rect* rect, double* mfcc) {
    int cell_width = (rect->w - 40) / MFCC_HISTORY;
    int cell_height = (rect->h - 35) / NUM_MFCC;
    
    double max_val = 0;
    for (int i = 0; i < NUM_MFCC * MFCC_HISTORY; i++) {
        if (fabs(mfcc_config.history[i]) > max_val) max_val = fabs(mfcc_config.history[i]);
    }
    if (max_val == 0) max_val = 1;
    
    for (int i = 0; i < NUM_MFCC; i++) {
        for (int t = 0; t < MFCC_HISTORY; t++) {
            int history_idx = (mfcc_config.history_pos - MFCC_HISTORY + t + MFCC_HISTORY) % MFCC_HISTORY;
            double value = mfcc_config.history[history_idx * NUM_MFCC + i];
            SDL_Color color;
            get_color_for_value(value, max_val, &color);
            SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
            
            SDL_Rect cell = {
                rect->x + 35 + t * cell_width, // Offset by 35 for y-axis labels
                rect->y + i * cell_height,
                cell_width,
                cell_height
            };
            SDL_RenderFillRect(renderer, &cell);
        }
    }
    
    SDL_Color text_color = {TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, 255};
    
    // Draw y-axis labels (MFCC coefficient indices)
    for (int i = 0; i < NUM_MFCC; i++) {
        char label[10];
        sprintf(label, "%d", i);
        render_text(renderer, label, rect->x + 5, rect->y + i * cell_height, text_color);
    }
    
    // Draw axis titles
    render_text(renderer, "Time (frames)", rect->x + rect->w/2 - 40, rect->y + rect->h - 30, text_color);
    render_text(renderer, "MFCC Coeff", rect->x + 5, rect->y + rect->h/2 - 10, text_color);
}

void cleanup_mfcc() {
    if (mfcc_config.mel_filters) {
        free(mfcc_config.mel_filters);
    }
    if (mfcc_config.history) {
        free(mfcc_config.history);
    }
}

void init_plots() {
    plots[0] = (PlotArea){NULL, {0}, "Time Domain"};
    plots[1] = (PlotArea){NULL, {0}, "Magnitude"};
    plots[2] = (PlotArea){NULL, {0}, "Phase"};
    plots[3] = (PlotArea){NULL, {0}, "Real"};
    plots[4] = (PlotArea){NULL, {0}, "Imaginary"};
    plots[5] = (PlotArea){NULL, {0}, "MFCC"};
    
    // Calculate window size for 2x3 grid
    int grid_cols = 2;
    int grid_rows = 3;
    int window_width = grid_cols * (PLOT_WIDTH + WINDOW_PADDING) - WINDOW_PADDING;
    int window_height = grid_rows * (PLOT_HEIGHT + WINDOW_PADDING) - WINDOW_PADDING;
    
    // Get screen size
    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    int screen_width = DM.w;
    int screen_height = DM.h;
    
    // If window is too tall, scale down plot height and padding
    if (window_height > screen_height - 80) { // leave some margin for taskbar etc
        double scale = (double)(screen_height - 80) / window_height;
        PLOT_HEIGHT = (int)(PLOT_HEIGHT * scale);
        WINDOW_PADDING = (int)(WINDOW_PADDING * scale);
        window_height = grid_rows * (PLOT_HEIGHT + WINDOW_PADDING) - WINDOW_PADDING;
        window_width = grid_cols * (PLOT_WIDTH + WINDOW_PADDING) - WINDOW_PADDING;
    }
    // If window is too wide, scale down plot width and padding
    if (window_width > screen_width - 80) {
        double scale = (double)(screen_width - 80) / window_width;
        PLOT_WIDTH = (int)(PLOT_WIDTH * scale);
        WINDOW_PADDING = (int)(WINDOW_PADDING * scale);
        window_height = grid_rows * (PLOT_HEIGHT + WINDOW_PADDING) - WINDOW_PADDING;
        window_width = grid_cols * (PLOT_WIDTH + WINDOW_PADDING) - WINDOW_PADDING;
    }
    
    // Create main window
    main_window = SDL_CreateWindow("Fourier Transform Visualization",
                                 SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                 window_width, window_height,
                                 SDL_WINDOW_SHOWN);
    if (!main_window) {
        printf("Failed to create main window\n");
        return;
    }
    
    main_renderer = SDL_CreateRenderer(main_window, -1, SDL_RENDERER_ACCELERATED);
    if (!main_renderer) {
        SDL_DestroyWindow(main_window);
        main_window = NULL;
        printf("Failed to create main renderer\n");
        return;
    }
    
    // Initialize plot areas
    for (int i = 0; i < 6; i++) {
        int row = i / 2;
        int col = i % 2;
        plots[i].rect.x = col * (PLOT_WIDTH + WINDOW_PADDING);
        plots[i].rect.y = row * (PLOT_HEIGHT + WINDOW_PADDING);
        plots[i].rect.w = PLOT_WIDTH;
        plots[i].rect.h = PLOT_HEIGHT;
        plots[i].renderer = main_renderer;
    }
}

void cleanup_plots() {
    if (main_renderer) {
        SDL_DestroyRenderer(main_renderer);
    }
    if (main_window) {
        SDL_DestroyWindow(main_window);
    }
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO) != 0) return 1;
    if (TTF_Init() != 0) {
        SDL_Quit();
        return 1;
    }

    font = TTF_OpenFont("arial.ttf", FONT_SIZE);
    if (!font) {
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    init_signal_config();
    init_plots();
    init_audio_buffer();
    init_mfcc();

    SDL_AudioSpec want, have;
    SDL_memset(&want, 0, sizeof(want));
    want.freq = SAMPLE_RATE;
    want.format = AUDIO_F32;
    want.channels = 1;
    want.samples = 1024;
    want.callback = audio_callback;

    SDL_AudioDeviceID dev = SDL_OpenAudioDevice(NULL, 1, &want, &have, 0);
    if (dev == 0) {
        printf("Failed to open audio device: %s\n", SDL_GetError());
        cleanup_audio_buffer();
        cleanup_plots();
        cleanup_mfcc();
        TTF_CloseFont(font);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_PauseAudioDevice(dev, 0);

    double* input_signal = (double*)malloc(NUM_SAMPLES * sizeof(double));
    if (!input_signal) {
        SDL_CloseAudioDevice(dev);
        cleanup_audio_buffer();
        cleanup_plots();
        cleanup_mfcc();
        TTF_CloseFont(font);
        TTF_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Event event;
    while (is_running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) is_running = 0;
            handle_input(&event);
        }

        get_audio_data(input_signal, NUM_SAMPLES);
        double complex* dft_result = compute_dft(input_signal, NUM_SAMPLES);
        if (!dft_result) break;

        double* magnitude = (double*)malloc(NUM_SAMPLES * sizeof(double));
        double* phase = (double*)malloc(NUM_SAMPLES * sizeof(double));
        double* real = (double*)malloc(NUM_SAMPLES * sizeof(double));
        double* imag = (double*)malloc(NUM_SAMPLES * sizeof(double));

        for (int i = 0; i < NUM_SAMPLES; i++) {
            magnitude[i] = cabs(dft_result[i]);
            phase[i] = carg(dft_result[i]);
            real[i] = creal(dft_result[i]);
            imag[i] = cimag(dft_result[i]);
        }

        // Clear the main window
        SDL_SetRenderDrawColor(main_renderer, 0, 0, 0, 255);
        SDL_RenderClear(main_renderer);

        // Time domain plot
        if (plots[0].renderer) {
            draw_plot_area(main_renderer, &plots[0].rect);
            draw_grid(main_renderer, &plots[0].rect, plots[0].rect.w, plots[0].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[0].rect, plots[0].rect.w, plots[0].rect.h - 35, 8, 6);
            plot_function(main_renderer, &plots[0].rect, input_signal, NUM_SAMPLES, plots[0].rect.w, plots[0].rect.h - 35, colors.time);
            draw_axis_numbers(main_renderer, &plots[0].rect, plots[0].rect.w, plots[0].rect.h - 35, 8, 6, 1.0, 1.0, 0);
            draw_controls(main_renderer, &plots[0].rect);
        }

        // Magnitude plot
        if (plots[1].renderer && show_magnitude) {
            draw_plot_area(main_renderer, &plots[1].rect);
            draw_grid(main_renderer, &plots[1].rect, plots[1].rect.w, plots[1].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[1].rect, plots[1].rect.w, plots[1].rect.h - 35, 8, 6);
            plot_function(main_renderer, &plots[1].rect, magnitude, NUM_SAMPLES/2, plots[1].rect.w, plots[1].rect.h - 35, colors.magnitude);
            draw_axis_numbers(main_renderer, &plots[1].rect, plots[1].rect.w, plots[1].rect.h - 35, 8, 6, SAMPLE_RATE/2, 1.0, 1);
        }

        // Phase plot
        if (plots[2].renderer && show_phase) {
            draw_plot_area(main_renderer, &plots[2].rect);
            draw_grid(main_renderer, &plots[2].rect, plots[2].rect.w, plots[2].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[2].rect, plots[2].rect.w, plots[2].rect.h - 35, 8, 6);
            plot_function(main_renderer, &plots[2].rect, phase, NUM_SAMPLES/2, plots[2].rect.w, plots[2].rect.h - 35, colors.phase);
            draw_axis_numbers(main_renderer, &plots[2].rect, plots[2].rect.w, plots[2].rect.h - 35, 8, 6, SAMPLE_RATE/2, M_PI, 1);
        }

        // Real plot
        if (plots[3].renderer && show_real) {
            draw_plot_area(main_renderer, &plots[3].rect);
            draw_grid(main_renderer, &plots[3].rect, plots[3].rect.w, plots[3].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[3].rect, plots[3].rect.w, plots[3].rect.h - 35, 8, 6);
            plot_function(main_renderer, &plots[3].rect, real, NUM_SAMPLES/2, plots[3].rect.w, plots[3].rect.h - 35, colors.real);
            draw_axis_numbers(main_renderer, &plots[3].rect, plots[3].rect.w, plots[3].rect.h - 35, 8, 6, SAMPLE_RATE/2, 1.0, 1);
        }

        // Imaginary plot
        if (plots[4].renderer && show_imaginary) {
            draw_plot_area(main_renderer, &plots[4].rect);
            draw_grid(main_renderer, &plots[4].rect, plots[4].rect.w, plots[4].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[4].rect, plots[4].rect.w, plots[4].rect.h - 35, 8, 6);
            plot_function(main_renderer, &plots[4].rect, imag, NUM_SAMPLES/2, plots[4].rect.w, plots[4].rect.h - 35, colors.imaginary);
            draw_axis_numbers(main_renderer, &plots[4].rect, plots[4].rect.w, plots[4].rect.h - 35, 8, 6, SAMPLE_RATE/2, 1.0, 1);
        }

        // MFCC plot
        if (plots[5].renderer) {
            draw_plot_area(main_renderer, &plots[5].rect);
            draw_grid(main_renderer, &plots[5].rect, plots[5].rect.w, plots[5].rect.h - 35, 8, 6);
            draw_ticks(main_renderer, &plots[5].rect, plots[5].rect.w, plots[5].rect.h - 35, 8, 6);
            double* mfcc = (double*)malloc(NUM_MFCC * sizeof(double));
            compute_mfcc(magnitude, mfcc);
            draw_mfcc_heatmap(main_renderer, &plots[5].rect, mfcc);
            SDL_Color text_color = {TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, 255};
            render_text(main_renderer, "MFCC Coefficients", plots[5].rect.x + 5, plots[5].rect.y + plots[5].rect.h - 30, text_color);
            render_text(main_renderer, "Time", plots[5].rect.x + plots[5].rect.w - 40, plots[5].rect.y + plots[5].rect.h - 30, text_color);
            free(mfcc);
        }

        SDL_RenderPresent(main_renderer);

        free(dft_result);
        free(magnitude);
        free(phase);
        free(real);
        free(imag);

        SDL_Delay(16);
    }

    SDL_CloseAudioDevice(dev);
    free(input_signal);
    cleanup_audio_buffer();
    cleanup_plots();
    cleanup_mfcc();
    TTF_CloseFont(font);
    TTF_Quit();
    SDL_Quit();

    return 0;
} 