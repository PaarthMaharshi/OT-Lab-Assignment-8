#!/bin/bash
# Build the patched xv6 on Windows/MSYS2 (Track G). Usage: bash build.sh
# The project path contains a space, which breaks make, so the build happens in a space-free copy.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)/xv6-riscv"
B="${XV6_BUILD:-/c/Users/pawan/AppData/Local/Temp/xv6build}"
rm -rf "$B" && mkdir -p "$B/tmp" && cp -r "$SRC/." "$B/src" && rm -rf "$B/src/.git"
W=$(cygpath -w "$B/tmp"); export TMP="$W" TEMP="$W" TMPDIR="$W"
cd "$B/src"
# 1. host tool mkfs with the ucrt64 gcc (kept off PATH below: its DLLs clash with the clang64 cross compiler)
PATH=/c/msys64/ucrt64/bin:/c/msys64/usr/bin gcc -Wno-unknown-attributes -I. -o mkfs/mkfs mkfs/mkfs.c
# 2. kernel, user programs and the file system image with the clang64 RISC-V toolchain
export PATH=/c/msys64/clang64/bin:/c/msys64/usr/bin:/c/Windows/System32   # explicit: ucrt64 must not be on PATH
# -pipe: GCC under MSYS make otherwise tries to write temp files to C:\WINDOWS
make -j8 TOOLPREFIX=riscv64-unknown-elf- CC="riscv64-unknown-elf-gcc -pipe" kernel/kernel fs.img
echo "built in $B/src"
