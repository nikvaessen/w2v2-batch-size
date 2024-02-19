# make dir if not exist
mkdir -p csv

# fix all csvs
for file in raw/*; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        # Add your processing commands here
        python fix_df.py --in "$file" --out csv/"$(basename "$file")"
    fi
done
