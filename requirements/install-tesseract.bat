mkdir F:\sudoku-solver\requirements\Projects\source
mkdir F:\sudoku-solver\requirements\Projects\win64
SET INSTALL_DIR=F:\sudoku-solver\requirements\Projects\win64
SET PATH=%PATH%;%INSTALL_DIR%\bin;
chdir F:\sudoku-solver\requirements\Projects\source

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" x64

git clone --depth 1 https://github.com/zlib-ng/zlib-ng.git
chdir zlib-ng
cmake -Bbuild -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DBUILD_SHARED_LIBS=OFF -DZLIB_COMPAT=ON -DZLIB_ENABLE_TESTS=OFF -DINSTALL_UTILS=OFF
cmake --build build --config Release --target install
chdir ..

curl -sSL https://download.sourceforge.net/libpng/lpng1640.zip -o lpng1640.zip
unzip.exe -qq lpng1640.zip
chdir lpng1640
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DPNG_TESTS=OFF -DPNG_SHARED=OFF
cmake --build build --config Release --target install
chdir ..

curl -sSL  https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/win64/nasm-2.16.01-win64.zip -o nasm-2.16.01-win64.zip
unzip -qq  nasm-2.16.01-win64.zip
copy nasm-2.16.01\*.exe %INSTALL_DIR%\bin\

git clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo.git
chdir libjpeg-turbo
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DWITH_TURBOJPEG=OFF
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://gitlab.com/libtiff/libtiff.git
chdir libtiff
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -Dtiff-tools=OFF -Dtiff-tests=OFF -Dtiff-contrib=OFF -Dtiff-docs=OFF
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://github.com/zdenop/jbigkit.git
chdir jbigkit
cmake -Bbuild -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DBUILD_PROGRAMS=OFF -DBUILD_TOOLS=OFF -DCMAKE_WARN_DEPRECATED=OFF
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://github.com/facebook/zstd.git
chdir zstd\build\cmake
cmake -Bbuild -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% 
cmake --build build --config Release --target install
chdir ..\..\..


git clone --depth 1 https://github.com/tukaani-project/xz.git
cd xz
cmake -Bbuild -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://github.com/xbmc/giflib.git
chdir giflib
cmake -Bbuild -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% 
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://chromium.googlesource.com/webm/libwebp
chdir libwebp
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DWEBP_BUILD_WEBP_JS=OFF -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_WEBP_JS=OFF
cmake --build build --config Release --target install
chdir ..

git clone --depth 1 https://github.com/uclouvain/openjpeg.git
chdir openjpeg
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% 
cmake --build build --config Release --target install
chdir ..

git clone --depth=1 https://github.com/DanBloomberg/leptonica
chdir leptonica
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DSW_BUILD=OFF -DBUILD_PROG=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_PROG=ON
cmake --build build --config Release --target install
chdir ..

git clone https://github.com/tesseract-ocr/tesseract tesseract
chdir tesseract
cmake -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=%INSTALL_DIR% -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% -DSW_BUILD=OFF -DBUILD_SHARED_LIBS=ON -DENABLE_LTO=ON -DBUILD_TRAINING_TOOLS=ON -DFAST_FLOAT=ON -DGRAPHICS_DISABLED=ON -DOPENMP_BUILD=OFF
cmake --build build --config Release --target install
chdir ..