python mainDisc.py --source Pamap2 --target Dsads --trainClf --trials 10 > ../results/Disc_pam_dsa.txt
python mainDisc.py --source Pamap2 --target Ucihar --trials 10 >  ../results/Disc_pam_uci.txt
python mainDisc.py --source Pamap2 --target Uschad --trials 10 >  ../results/Disc_pam_usc.txt
python mainDisc.py --source Dsads --target Pamap2 --trainClf --trials 10 >  ../results/Disc_dsa_pam.txt
python mainDisc.py --source Dsads --target Ucihar --trials 10 >  ../results/Disc_dsa_uci.txt
python mainDisc.py --source Dsads --target Uschad --trials 10 >  ../results/Disc_dsa_usc.txt
python mainDisc.py --source Ucihar --target Dsads --trainClf --trials 10 >  ../results/Disc_uci_dsa.txt
python mainDisc.py --source Ucihar --target Pamap2 --trials 10 >  ../results/Disc_uci_pam.txt
python mainDisc.py --source Ucihar --target Uschad --trials 10 >  ../results/Disc_uci_usc.txt
python mainDisc.py --source Uschad --target Dsads --trainClf --trials 10 >  ../results/Disc_usc_dsa.txt
python mainDisc.py --source Uschad --target Ucihar --trials 10 >  ../results/Disc_usc_uci.txt
python mainDisc.py --source Uschad --target Pamap2 --trials 10 >  ../results/Disc_usc_pam.txt

python mainDisc.py --TLParamsFile asymParams.json --source Pamap2 --target Dsads --trainClf --trials 10 > ../results/Disc_pam_dsa.txt
python mainDisc.py --TLParamsFile asymParams.json --source Pamap2 --target Ucihar --trials 10 >  ../results/Disc_pam_uci.txt
python mainDisc.py --TLParamsFile asymParams.json --source Pamap2 --target Uschad --trials 10 >  ../results/Disc_pam_usc.txt
python mainDisc.py --TLParamsFile asymParams.json --source Dsads --target Pamap2 --trainClf --trials 10 >  ../results/Disc_dsa_pam.txt
python mainDisc.py --TLParamsFile asymParams.json --source Dsads --target Ucihar --trials 10 >  ../results/Disc_dsa_uci.txt
python mainDisc.py --TLParamsFile asymParams.json --source Dsads --target Uschad --trials 10 >  ../results/Disc_dsa_usc.txt
python mainDisc.py --TLParamsFile asymParams.json --source Ucihar --target Dsads --trainClf --trials 10 >  ../results/Disc_uci_dsa.txt
python mainDisc.py --TLParamsFile asymParams.json --source Ucihar --target Pamap2 --trials 10 >  ../results/Disc_uci_pam.txt
python mainDisc.py --TLParamsFile asymParams.json --source Ucihar --target Uschad --trials 10 >  ../results/Disc_uci_usc.txt
python mainDisc.py --TLParamsFile asymParams.json --source Uschad --target Dsads --trainClf --trials 10 >  ../results/Disc_usc_dsa.txt
python mainDisc.py --TLParamsFile asymParams.json --source Uschad --target Ucihar --trials 10 >  ../results/Disc_usc_uci.txt
python mainDisc.py --TLParamsFile asymParams.json --source Uschad --target Pamap2 --trials 10 >  ../results/Disc_usc_pam.txt

python mainSL.py --source Pamap2 --target Dsads > ../results/softLab_pam_dsa.txt
python mainSL.py --source Pamap2 --target Ucihar > ../results/softLab_pam_uci.txt
python mainSL.py --source Pamap2 --target Uschad > ../results/softLab_pam_usc.txt
python mainSL.py --source Dsads --target Pamap2 > ../results/softLab_dsa_pam.txt
python mainSL.py --source Dsads --target Ucihar > ../results/softLab_dsa_uci.txt
python mainSL.py --source Dsads --target Uschad > ../results/softLab_dsa_usc.txt
python mainSL.py --source Ucihar --target Pamap2 > ../results/softLab_uci_pam.txt
python mainSL.py --source Ucihar --target Dsads > ../results/softLab_uci_dsa.txt
python mainSL.py --source Ucihar --target Uschad > ../results/softLab_uci_usc.txt
python mainSL.py --source Uschad --target Pamap2 > ../results/softLab_usc_pam.txt
python mainSL.py --source Uschad --target Dsads > ../results/softLab_usc_dsa.txt
python mainSL.py --source Uschad --target Ucihar > ../results/softLab_usc_uci.txt


#- [x]  Experimentos com essa arquitetura, variando o alfa. (5x cada modelo A→B) para o metodo features simetricas
#- [x]  Rodar 5x experimentos de softLabel.
#- [x]  Rodar 5x experimentos de features assimétricas.
