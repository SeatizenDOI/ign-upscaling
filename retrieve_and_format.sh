#!/bin/bash


## Download all compressed file from IGN.

OUTPUT_FOLDER=./data/ign/raw_data
mkdir -p $OUTPUT_FOLDER

# echo "Downloading IGN 2022 layer part 1."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.001 \
# --output ${OUTPUT_FOLDER}/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.001

# echo "Downloading IGN 2022 layer part 2."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.002 \
# --output ${OUTPUT_FOLDER}/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.002

# echo "Downloading IGN 2022 layer part 3."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.003 \
# --output ${OUTPUT_FOLDER}/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.003

# echo "Downloading IGN 2017 layer part 1."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.001 \
# --output ${OUTPUT_FOLDER}/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.001

# echo "Downloading IGN 2017 layer part 2."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.002 \
# --output ${OUTPUT_FOLDER}/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.002

# echo "Downloading IGN 2017 layer part 3."

# curl https://data.geopf.fr/telechargement/download/BDORTHO/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.003 \
# --output ${OUTPUT_FOLDER}/BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.003

## Extract compressed file.

cd $OUTPUT_FOLDER

# echo "Extracting 2022."
# 7z x BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01.7z.001

# echo "Extracting 2017."
# 7z x BDORTHO_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01.7z.001


declare -a layer_to_keep=(
    "0310-7670" "0310-7675"
    "0315-7665" "0315-7670" "0315-7675"
    "0320-7655" "0320-7660" "0320-7665"
    "0325-7650" "0335-7640" "0340-7640"
)

## Transform JP2 file to COG file to easily vizualize them into QGIS.
cd ..

# FOLDER_INPUT_2022=./raw_data/BDORTHO_2-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2022-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2024-03-00066/OHR_RVB_0M20_JP2-E080_RGR92UTM40S_D974-2022
# FOLDER_OUTPUT_2022=IGN_2022_974

# rm -rf $FOLDER_OUTPUT_2022
# mkdir -p $FOLDER_OUTPUT_2022

# echo "Transforming 2022."

# for layer in "${layer_to_keep[@]}"
# do
#     file_input="${FOLDER_INPUT_2022}/974-2022-${layer}-U40S-0M20-E080.jp2"
#     basename=$(basename "$file_input" .jp2)
#     file_output="${FOLDER_OUTPUT_2022}/${basename}.tif"

#     echo "Transform ${file_input} to ${file_output} as COG."

#     gdal_translate -of COG $file_input $file_output

# done


FOLDER_INPUT_2017=./raw_data/ORTHOHR_1-0_RVB-0M20_JP2-E080_RGR92UTM40S_D974_2017-01-01/ORTHOHR/1_DONNEES_LIVRAISON_2019-09-00386/OHR_RVB_0M20_JP2-E080_RGR92UTM40S_D974-2017
FOLDER_OUTPUT_2017=IGN_2017_974

rm -rf $FOLDER_OUTPUT_2017
mkdir -p $FOLDER_OUTPUT_2017

echo "Transforming 2017."

for layer in "${layer_to_keep[@]}"
do
    file_input="${FOLDER_INPUT_2017}/974-2017-${layer}-U40S-0M20-E080.jp2"
    basename=$(basename "$file_input" .jp2)
    file_mask="${FOLDER_OUTPUT_2017}/${basename}_mask.tif"
    file_output="${FOLDER_OUTPUT_2017}/${basename}.tif"

    echo "Transform ${file_input} to ${file_output} as COG."
    gdalwarp -cutline ../emprise_lagoon.geojson -crop_to_cutline -dstalpha "$file_input" "$file_mask"
    gdal_translate -of COG $file_mask $file_output

done