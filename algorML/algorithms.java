
package algorML;

import java.util.Arrays;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.clusterers.SimpleKMeans;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Normalize;


 public class algorithms{
         
   Instances  dane;
   Instances trainDane;
   Instances testDane;
   
   //---------------------------------------
   // KONSTRUKTOR DLA KLASY algorithms
   //---------------------------------------
 public algorithms() throws Exception{ 
    dane = ConverterUtils.DataSource.read("C:/Program Files/Weka-3-9/data/weather.numeric.arff");
  //     dane = ConverterUtils.DataSource.read("M:/Kasia/PROGRAMOWANIE/PYTHON/Projekt Machine Learning/DANE/heartSkr.arff");
  //  dane = ConverterUtils.DataSource.read("M:/Kasia/PROGRAMOWANIE/PYTHON/Projekt Machine Learning/DANE/DBC1.arff");
  //  dane = ConverterUtils.DataSource.read("M:/Kasia/PROGRAMOWANIE/PYTHON/Projekt Machine Learning/DANE/dbcProbka.arff");
   
   
    //Usuwanie atrybutów
    dane.deleteAttributeAt(1);
    System.out.println("ilość atrybutów po usunięciu: "+dane.numAttributes());
    
   
    
    
    
   //rozmiar danych
   double size = 30;
   
   int trainSize = (int) Math.round(dane.numInstances() * size/100);
   int testSize = dane.numInstances() - trainSize;
   
   trainDane = new Instances(dane, 0, trainSize);
   testDane = new Instances(dane, trainSize, testSize);
   
   dane.setClassIndex(dane.numAttributes()-1);
   trainDane.setClassIndex(trainDane.numAttributes()-1);
   testDane.setClassIndex(testDane.numAttributes()- 1);
   
   
   
   
  }


    public Instance weryfikacja(double... w) throws Exception {
        double[] wExt = Arrays.copyOf(w, w.length+1);   
        Instance wiersz = new DenseInstance(1, wExt);
        System.out.println("wiersz wersja 0: "+wiersz);
        return wiersz;
    }
    
    
        
     // -----------------------------------------------------
     // BUDOWA KLASYFIKATORA  Naive Bayes
     // -----------------------------------------------------
     
     NaiveBayes  KlNb(Instances dane) throws Exception{
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
     
     J48  drzewaJ48(Instances dane) throws Exception{
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
     
     MultilayerPerceptron sieciNeuronowe(Instances dane) throws Exception {
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
    

     IBk algokNN(Instances dane) throws Exception {
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
  
    
     
     
    
    
     
 }