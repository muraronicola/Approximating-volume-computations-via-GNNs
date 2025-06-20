#!/bin/bash
#8698
path_configurations="./configurations/custom/normalization/"

configurations_path=("normalization_m_3_r_2/" "normalization_m_4_r_2/" "normalization_m_4_r_3/" "normalization_m_5_r_3/" "normalization_m_5_r_4/")
file_name=("conf_1" "conf_2")


for i in ${!configurations_path[@]}; do
    for j in ${!file_name[@]}; do
        echo "Training with configuration: ${configurations_path[$i]}${file_name[$j]}"

        path="${path_configurations}${configurations_path[$i]}${file_name[$j]}.json"

        time python3 ./main.py --path_configuration ${path}
        echo ""
        echo "----------------------------------------"
        echo ""
        echo ""
    done
done







path_configurations="./configurations/custom/generalization_v1/"
configurations=("generalization_m_3_r_2_to_m_4_r_2" "generalization_m_3_r_2_to_m_4_r_3" "generalization_m_4_r_2_to_m_4_r_3" "generalization_m_4_r_3_to_m_5_r_3" "generalization_m_5_r_3_to_m_5_r_4")



for i in ${!configurations[@]}; do
    echo "Training with configuration: ${configurations[$i]}"

    path="${path_configurations}${configurations[$i]}.json"

    time python3 ./main.py --path_configuration ${path}
    echo ""
    echo "----------------------------------------"
    echo ""
    echo ""
done


path_configurations="./configurations/custom/generalization_v2/"
configurations=("generalization_m_3_r_2_to_m_4_r_2" "generalization_m_3_r_2_to_m_4_r_3" "generalization_m_4_r_2_to_m_4_r_3" "generalization_m_4_r_3_to_m_5_r_3" "generalization_m_5_r_3_to_m_5_r_4")



for i in ${!configurations[@]}; do
    echo "Training with configuration: ${configurations[$i]}"

    path="${path_configurations}${configurations[$i]}.json"

    time python3 ./main.py --path_configuration ${path}
    echo ""
    echo "----------------------------------------"
    echo ""
    echo ""
done



path_configurations="./configurations/custom/generalization_v3/"
configurations=("generalization_m_3_r_2_to_m_4_r_2" "generalization_m_3_r_2_to_m_4_r_3" "generalization_m_4_r_2_to_m_4_r_3" "generalization_m_4_r_3_to_m_5_r_3" "generalization_m_5_r_3_to_m_5_r_4")



for i in ${!configurations[@]}; do
    echo "Training with configuration: ${configurations[$i]}"

    path="${path_configurations}${configurations[$i]}.json"

    time python3 ./main.py --path_configuration ${path}
    echo ""
    echo "----------------------------------------"
    echo ""
    echo ""
done




path_configurations="./configurations/custom/all_data/"
path="${path_configurations}all_data.json"
time python3 ./main.py --path_configuration ${path}
echo ""
echo "----------------------------------------"
echo ""
echo ""