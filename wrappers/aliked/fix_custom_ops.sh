FILE1="wrappers/aliked/aliked/custom_ops/get_patches_cuda.cu"
FILE2="wrappers/aliked/aliked/custom_ops/get_patches.cpp"  # if it exists

# backups
cp "$FILE1" "$FILE1.bak"; [ -f "$FILE2" ] && cp "$FILE2" "$FILE2.bak"

# ensure the macro uses x.is_cuda()
sed -i 's/#define CHECK_CUDA(x).*/#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")/' "$FILE1"

# clean up any accidental replacements like input.scalar_type().is_cuda()
sed -i 's/\.scalar_type().is_cuda()/\.is_cuda()/g' "$FILE1" "$FILE2"

# build
cd wrappers/aliked/aliked/custom_ops && sh build.sh
