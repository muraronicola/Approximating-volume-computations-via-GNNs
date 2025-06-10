#!/bin/bash
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




#path_configurations="./configurations/custom/all_data/"
#path="${path_configurations}all_data.json"
#time python3 ./main.py --path_configuration ${path}
#echo ""
#echo "----------------------------------------"
#echo ""
#echo ""