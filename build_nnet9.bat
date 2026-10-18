@echo off
rem ==========================================================================
rem  ChesslabNNet9 - build the exe with the 533M 768 net and the book built in
rem
rem  Put this .bat, ChesslabNNet9.c, k7net.bin (the 533M / 35-epoch net)
rem  and k7Opennings.bin together in one folder (D:\CLCoreGCC\ChesslabNNet9)
rem  and double-click it, or run it from PowerShell:   .\build_nnet9.bat
rem  Uses MSYS2 MinGW64 gcc.
rem ==========================================================================
setlocal
cd /d "%~dp0"

where gcc >nul 2>nul || set "PATH=D:\mingw64\bin;%PATH%"
where gcc >nul 2>nul || (
    echo gcc not found. Install MSYS2 MinGW64 or add its bin folder to PATH.
    pause
    exit /b 1
)

set MISSING=
for %%f in (ChesslabNNet9.c k7net.bin k7Opennings.bin) do (
    if not exist "%%f" (
        echo   missing: %%f
        set MISSING=1
    )
)
if defined MISSING (
    echo.
    echo Every file above must be in %CD% before building.
    echo ^(Net has another name? Rename it to k7net.bin, the 533M 768 net.^)
    pause
    exit /b 1
)

echo Building ChesslabNNet9.exe ...
gcc -O3 -march=native -static -Wa,-muse-unaligned-vector-move -o ChesslabNNet9.exe ChesslabNNet9.c -lm
if errorlevel 1 goto :fail

echo.
echo Done: ChesslabNNet9.exe  (net and book are inside - ship only the exe)
echo   Arena options: OwnBook, BookPick Best/Weighted, BookMaxPly, BookFile,
echo   NetFile ^<embedded^>. Turn OwnBook off for engine-vs-engine testing.
echo.
echo Quick check (perft 5 must be 4865609, bench 9 must be 536905 nodes):
(echo isready& echo perft 5& echo bench 9& echo quit) | ChesslabNNet9.exe
echo.
pause
exit /b 0

:fail
echo Build failed - see the messages above.
pause
exit /b 1
