$fontUrl = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
Invoke-WebRequest -Uri $fontUrl -OutFile "arial.ttf"
Write-Host "Font downloaded successfully!" 