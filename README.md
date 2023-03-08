All'interno di questo repository sono presenti tutti gli applicativi realizzati per la tesi triennale in ingegneria e scienze informatiche (UniBo, sede di Cesena) illustrata nel file tesi_t_lorenzo_zanetti5.pdf.



Come prima cosa nella cartella docker è presente un docker file che inizializza un immagine utile a lanciare gli applicativa, si consiglia di creare l'immagine andando nella directory appena citata e lanciando il comando:
docker build -t radiography_retrieval .



Una volta fatto questo gli applicativi utilizzabili sono: 
-train.py, per l'addestramento iniziale del modello;
-metric_learning.py, per l'addestramento del modello di metric learning, partendo dal modello già addestrato;
-test_retrieval.py, per valutare le metriche in task di retrieval del modello di deep metric learning;
-web_app.py, che apre un applicativo web in cui vengono mostrate le capacità del modello;
-explainability_integrated_gradients.py, che mostra l'analisi con tecniche di explainability di un singolo record indicato;



Questi applicativi possono direttamente essere utilizzati all'interno dell'immagine docker creata usando i parametri descritti al loro interno (visualizzabili utilizzando il parametro -h). Per semplicità sono stati creati piccoli script .sh che si occupato di creare l'ambiente docker e lanciare gli applicativi (in questo caso, per modificare i parametri, vanno modificati rispettivamente i file train.sh, metric_learning.sh, retrieval.sh, web_app_args.sh):
train_inside_docker.sh (per addestramento del modello), start_metric_learning.sh (per addestramento della loss del metric learning), test_retrieval.sh (per valutare il modello in task di retrieval) e start_web_app.sh (per lanciare l'applicativo web).



Per riprodurre i risultati ottenuti nella tesi è sufficiente riprodurre gli addestramenti utilizzando gli iperparametri descritti al capitolo 4 (sugli esperimenti condotti).