# python Drain_demo.py ../logs/HDFS/ paper3_Drain_result/ HDFS.log >> paper3_res.txt 
# python Spell_demo.py ../logs/HDFS/ paper3_Spell_result/ HDFS.log >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result/ HDFS.log >> paper3_res.txt 
# python SHISO_demo.py ../logs/HDFS/ paper3_SHISO_result/ HDFS.log >> paper3_res.txt 
# python IPLoM_demo.py ../logs/HDFS/ paper3_IPLoM_result/ HDFS.log >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau07/ HDFS.log 0.7 >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau075/ HDFS.log 0.75 >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau08/ HDFS.log 0.8 >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau085/ HDFS.log 0.85 >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau09/ HDFS.log 0.9 >> paper3_res.txt 
# python Drain_plus_2_demo.py ../logs/HDFS/ paper3_Drain_2_result_tau095/ HDFS.log 0.95 >> paper3_res.txt 

python noise_data_preposses.py paper3_Drain_result 
python noise_data_preposses.py paper3_Spell_result &
python noise_data_preposses.py paper3_Drain_2_result &
python noise_data_preposses.py paper3_SHISO_result &
python noise_data_preposses.py paper3_IPLoM_result  &
python noise_data_preposses.py paper3_Drain_2_result_tau07  &
python noise_data_preposses.py paper3_Drain_2_result_tau075 &
python noise_data_preposses.py paper3_Drain_2_result_tau08 &
python noise_data_preposses.py paper3_Drain_2_result_tau085  &
python noise_data_preposses.py paper3_Drain_2_result_tau09 &
python noise_data_preposses.py paper3_Drain_2_result_tau095 &
python noise_data_preposses.py paper3_Drain_2_result_tau06  &
