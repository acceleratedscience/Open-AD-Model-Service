DIRS="gt4sd_common gt4sd_guacamol gt4sd_moler gt4sd_paccmann gt4sd_regression gt4sd_reinvent gt4sd_torch gt4sd_inference_all"

# Iterate over each directory
for dir in $DIRS; do
    echo "[i] Running lock for $dir"
    cd openad-model-inference/$dir && poetry lock
    cd - > /dev/null
done

echo "[i] Running lock for openad-model-inference"
poetry lock
