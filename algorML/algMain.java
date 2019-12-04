
package algorML;

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.clusterers.Clusterer;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

public class algMain {
       
    
    
    //METODA DLA DOWOLNEGO KLASYFIKATORA 
    public  static Classifier callAlgorithms(algorithms alg, Instances dane, Classifier klasyfikator) throws Exception{
       
        // POWOŁANY JUŻ ZOSTAŁ KLASYFIKATOR

        if(dane == alg.trainDane){
            System.out.println("Zbiór uczymy na danych treningowych i testujemy na danych testowych");
        dane = alg.testDane;}
        
        System.out.println("");
        System.out.println("");
                   
       
        // OCENA DZIAŁANIA KLASYFIKATORA --> ewaluacja + crosswalidacja
        // EWALUACJA
        System.out.println("Zanim ewaluacja wyswietl w0 ");
   //     System.out.println("w0 "+ dane.instance(0));
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
        

        return klasyfikator;
    }
    
   
 
    
    public static void main(String[] args) throws Exception {
        
        System.out.println("Witaj");
        System.out.println("--------------");
        System.out.println("");
        
        
        
        System.out.println("Powołanie obiektu klasy algorithms ");
        algorithms nb = new algorithms();
        System.out.println("");
        System.out.println("Czas na obróbkę danych:");
        System.out.println("Nakładamy Filtr: Nominalne na Binarne");
        System.out.println("");
        NominalToBinary ntb = new NominalToBinary();
        ntb.setInputFormat(nb.dane);
        nb.dane = Filter.useFilter(nb.dane, ntb);
        
        
      //Normalizacja   --   konieczna dla Multilayer Perceptron
        Normalize nor = new Normalize();
        nor.setInputFormat(nb.dane);
        nb.dane = Filter.useFilter(nb.dane, nor);
        
     //   System.out.println("");
    //    System.out.println("Dane:");
    //    System.out.println(nb.dane);
        System.out.println("");
        System.out.println("Numer indeksu klasowego:  "+nb.dane.classIndex());
        System.out.println("Nazwa indeksu klastrowego: "+nb.dane.classAttribute());
      //  System.out.println("NaiwnyBayes:  " + nb);
        
        
        
      //  callAlgorithms(nb, nb.trainDane, nb.KlNb(nb.trainDane));
      //  callAlgorithms(nb, nb.trainDane, nb.drzewaJ48(nb.trainDane));
        
      callAlgorithms(nb, nb.trainDane, nb.sieciNeuronowe(nb.trainDane));
        
     // callAlgorithms(nb, nb.trainDane, nb.algokNN(nb.trainDane));
     //  callAlgorithms(nb, nb.trainDane, nb.NaiwnyBayes(nb.trainDane)); 
       
      
        
    }
    
 
}
