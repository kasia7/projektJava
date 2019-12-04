
package algorML;

import static aMlAiDl.algMain.callAlgorithms;
import java.util.Arrays;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.Instances;


public class metodyAlg extends algorithms {

    algorithms alg = new algorithms();

    public metodyAlg() throws Exception {
    }

    
        
     // -----------------------------------------------------
     // BUDOWA KLASYFIKATORA  Naive Bayes
     // -----------------------------------------------------
     
     NaiveBayes  KlNb() throws Exception{
         // POWOŁUJEMY DO ŻYCIA OBIEKT  KlNb
         NaiveBayes algnb = new NaiveBayes(); 
                   
        //najpierw wyświetlenie opcji standardowych
         String[] opcje = algnb.getOptions(); //tablica napisów 
         System.out.println("Opcje przed zbudowaniem: "+Arrays.toString(opcje));
        
         
      // DODAJEMY OPCJE
      // drzewo.setMinNumObj(5);

       System.out.println("w0 "+dane.instance(0));
       System.out.println(" ");
       
       System.out.println("DANE:  trainDane");
       System.out.println(dane);
       System.out.println(" ");
       System.out.println(" ");
       
      // BUDUJEMY KLASYFIKATOR
       algnb.buildClassifier(dane);
       System.out.println("Budowa klasyfikatora Naive Bayes: ");
       System.out.println(algnb);
       System.out.println(" ");
        
       // WYŚWIETLENIE DODATKOWYCH OPCJI
        System.out.println("Opcje dodatkowe");
        System.out.println("");
         

System.out.println("Informacje o atrybucie klasowym: "+dane.classAttribute());
System.out.println("Indeks klasowy  "+ dane.classIndex() );
System.out.println("Ilosc wartosci atr klasowego  "+ dane.numClasses() );
System.out.println(" ");  
//System.out.println("w0 "+dane.instance(0));
System.out.println("***************");
System.out.println(" ");
// System.out.println(drzewo);
         return algnb;
     }
     
        
     // -----------------------------------------------------
     // BUDOWA KLASYFIKATORA  C 4.5
     // -----------------------------------------------------
     
     J48  drzewaJ48() throws Exception{
         // POWOŁUJEMY DO ŻYCIA OBIEKT DRZEWO
         J48 algdrzewo = new J48(); 
                   
        //najpierw wyświetlenie opcji standardowych
         String[] opcje = algdrzewo.getOptions(); //tablica napisów 
         System.out.println("Opcje przed zbudowaniem: "+Arrays.toString(opcje));
        
         
// DODAJEMY OPCJE
      //  drzewo.setMinNumObj(5);
// PRZYCINANIE DRZEW
     //  drzewo.setUnpruned(true);
    //   drzewo.setSubtreeRaising(true);
       algdrzewo.setConfidenceFactor((float) 0.35);
//       drzewo.setReducedErrorPruning(true);
//       drzewo.setNumFolds(3);
          

       System.out.println("w0 "+dane.instance(0));

System.out.println(" ");

// BUDUJEMY KLASYFIKATOR
       algdrzewo.buildClassifier(dane);
       System.out.println("Budowa klasyfikatora Decision Trees J48: ");
       System.out.println(algdrzewo);
       System.out.println(" ");
        
// WYŚWIETLENIE DODATKOWYCH OPCJI
        System.out.println("rozmiar drzewa: "+algdrzewo.measureTreeSize()); 
        System.out.println("ilość liści: "+ algdrzewo.measureNumLeaves());
        System.out.println("");
         

System.out.println("Inofmracje o atrybucie klasowym: "+dane.classAttribute());
System.out.println("Indeks klasowy  "+ dane.classIndex() );
System.out.println("Ilosc wartosci atr klasowego  "+ dane.numClasses() );
System.out.println(" ");  
//System.out.println("w0 "+dane.instance(0));
System.out.println("***************");
System.out.println(" ");
// System.out.println(drzewo);
         return algdrzewo;
     }
     
    
     // -----------------------------------------------------
     // BUDOWA KLASYFIKATORA  MULTILAYER PERCEPTRON
     // -----------------------------------------------------
     
     MultilayerPerceptron sieciNeuronowe() throws Exception {
     MultilayerPerceptron algmp = new MultilayerPerceptron();
     
     //najpierw wyświetlimy opcje standardowe
       String[] opcje = algmp.getOptions();
       System.out.println(Arrays.toString(opcje));
     
       
         System.out.println(" Nakładamy dodakowe opcje  ");
         
     // DODAJEMY OPCJE
        algmp.setHiddenLayers("o");
        algmp.setLearningRate(0.1);
        algmp.setMomentum(0.1);
      //  mp.setTrainingTime(2000);
        
        
     //Budujemy klasyfikator 
       algmp.buildClassifier(dane);
      
       System.out.println("");
       
         
     
         System.out.println("Klasyfikator sieć neuronowa  --   Zbudowany: \n"+algmp);
       
     return algmp;
     }
     
     
     
     
     // -----------------------------------------------------
     // BUDOWA KLASYFIKATORA  IBK -- k-najbliższych sąsiadów
     // -----------------------------------------------------
    

     IBk algokNN() throws Exception {
     IBk kNN = new IBk();
     
     //najpierw wyświetlimy opcje standardowe
       String[] opcje = kNN.getOptions();
       System.out.println(Arrays.toString(opcje));
     
         EuclideanDistance ed = new EuclideanDistance();
                 
       
     // DODAJEMY OPCJE
        kNN.setKNN(4);
        kNN.getNearestNeighbourSearchAlgorithm().setDistanceFunction(ed);  
        
     //Budujemy klasyfikator 
     kNN.buildClassifier(dane);
     
     
     return kNN;
 } 
  
    
     
  
    
      //METODA DLA DOWOLNEGO KLASYFIKATORA 
    public  metodyAlg(Classifier klasyfikator) throws Exception{
       
        // POWOŁANY JUŻ ZOSTAŁ KLASYFIKATOR
      
       
        // OCENA DZIAŁANIA KLASYFIKATORA --> ewaluacja + crosswalidacja
        // EWALUACJA
        System.out.println("Zanim ewaluacja wyswietl w0 ");
        System.out.println("w0 "+ dane.instance(0));
        System.out.println("");
        
        System.out.println("-------------------EWALUACJA--------------------");
        Evaluation ev = new Evaluation(dane);        
        
        ev.evaluateModel(klasyfikator,dane); 
        System.out.println(ev.toMatrixString("Macierz błędów:"));
        double[][] T= ev.confusionMatrix();
        double proc = (T[0][0]+T[1][1])/dane.numInstances();
        System.out.println("Procent trafień = "+(100*proc));
        System.out.println("");
        
        System.out.println("---------a teraz podsumowania1---------");
        System.out.println("ev.toSummaryString()  " + ev.toSummaryString());
        
        
        // CROSSWALIDACJA
        System.out.println("-------------------CROSSWALIDACJA--------------------");
        for(int i = 0; i < 1; i++){
            ev = new Evaluation(dane);
            ev.crossValidateModel(klasyfikator, dane, 10, new Random());
            System.out.println(ev.toMatrixString("Macierz Błędów"));
            T = ev.confusionMatrix();
            proc = T[0][0] + T[1][1];
            System.out.printf("Procent trafień: %5.2f%%\n",proc/dane.numInstances()*100);
        }
        System.out.println("");
        System.out.println("---------a teraz podsumowania2---------");
        System.out.println("ev.toSummaryString()  " + ev.toSummaryString());
        

       
    }
    
    
    
    
    
    
    
  
    
    
}
