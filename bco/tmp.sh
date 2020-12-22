cd models
rm *.pdf
python ../post_process.py jloss --last
cd ..
open models/*.pdf
