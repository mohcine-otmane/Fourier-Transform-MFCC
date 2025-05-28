$sdlVersion = "2.28.5"
$sdlImageVersion = "2.6.3"
$sdlTtfVersion = "2.20.2"
$downloadUrl = "https://github.com/libsdl-org/SDL/releases/download/release-$sdlVersion/SDL2-devel-$sdlVersion-VC.zip"
$sdlImageUrl = "https://github.com/libsdl-org/SDL_image/releases/download/release-$sdlImageVersion/SDL2_image-devel-$sdlImageVersion-VC.zip"
$sdlTtfUrl = "https://github.com/libsdl-org/SDL_ttf/releases/download/release-$sdlTtfVersion/SDL2_ttf-devel-$sdlTtfVersion-VC.zip"
$zipFile = "SDL2.zip"
$sdlImageZip = "SDL2_image.zip"
$sdlTtfZip = "SDL2_ttf.zip"
$sdlDir = "C:\SDL2"

Write-Host "Downloading SDL2..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile

Write-Host "Downloading SDL2_image..."
Invoke-WebRequest -Uri $sdlImageUrl -OutFile $sdlImageZip

Write-Host "Downloading SDL2_ttf..."
Invoke-WebRequest -Uri $sdlTtfUrl -OutFile $sdlTtfZip

Write-Host "Creating SDL2 directory..."
New-Item -ItemType Directory -Force -Path $sdlDir
New-Item -ItemType Directory -Force -Path "$sdlDir\lib"
New-Item -ItemType Directory -Force -Path "$sdlDir\include"

Write-Host "Extracting SDL2..."
Expand-Archive -Path $zipFile -DestinationPath "temp_sdl" -Force

Write-Host "Extracting SDL2_image..."
Expand-Archive -Path $sdlImageZip -DestinationPath "temp_sdl_image" -Force

Write-Host "Extracting SDL2_ttf..."
Expand-Archive -Path $sdlTtfZip -DestinationPath "temp_sdl_ttf" -Force

Write-Host "Copying SDL2 files..."
Copy-Item "temp_sdl\SDL2-$sdlVersion\lib\x64\*" -Destination "$sdlDir\lib" -Force
Copy-Item "temp_sdl\SDL2-$sdlVersion\include\*" -Destination "$sdlDir\include" -Force
Copy-Item "temp_sdl\SDL2-$sdlVersion\lib\x64\SDL2.dll" -Destination "." -Force

Write-Host "Copying SDL2_image files..."
Copy-Item "temp_sdl_image\SDL2_image-$sdlImageVersion\lib\x64\*" -Destination "$sdlDir\lib" -Force
Copy-Item "temp_sdl_image\SDL2_image-$sdlImageVersion\include\*" -Destination "$sdlDir\include" -Force
Copy-Item "temp_sdl_image\SDL2_image-$sdlImageVersion\lib\x64\SDL2_image.dll" -Destination "." -Force

Write-Host "Copying SDL2_ttf files..."
Copy-Item "temp_sdl_ttf\SDL2_ttf-$sdlTtfVersion\lib\x64\*" -Destination "$sdlDir\lib" -Force
Copy-Item "temp_sdl_ttf\SDL2_ttf-$sdlTtfVersion\include\*" -Destination "$sdlDir\include" -Force
Copy-Item "temp_sdl_ttf\SDL2_ttf-$sdlTtfVersion\lib\x64\SDL2_ttf.dll" -Destination "." -Force

Write-Host "Cleaning up..."
Remove-Item -Path $zipFile -Force
Remove-Item -Path $sdlImageZip -Force
Remove-Item -Path $sdlTtfZip -Force
Remove-Item -Path "temp_sdl" -Recurse -Force
Remove-Item -Path "temp_sdl_image" -Recurse -Force
Remove-Item -Path "temp_sdl_ttf" -Recurse -Force

Write-Host "SDL2, SDL2_image, and SDL2_ttf setup complete!" 