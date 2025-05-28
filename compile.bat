@echo off
gcc fourier_plot.c -o fourier_plot -I"C:\SDL2\include" -L"C:\SDL2\lib" -lSDL2main -lSDL2 -lSDL2_ttf -lm 