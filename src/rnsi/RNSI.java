/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rnsi;

import java.util.ArrayList;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizeArray;

/**
 *
 * @author beltraoluis
 */
public class RNSI {
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        /* 
        Dados de treinamento
        "sepal_l","sepal_w","petal_l","petal_w","species"
        5.1,3.5,1.4,0.2,Iris-setosa
        4.6,3.1,1.5,0.2,Iris-setosa
        5.0,3.6,1.4,0.2,Iris-setosa
        4.6,3.4,1.4,0.3,Iris-setosa
        5.0,3.4,1.5,0.2,Iris-setosa
        4.4,2.9,1.4,0.2,Iris-setosa
        4.9,3.1,1.5,0.1,Iris-setosa
        5.4,3.7,1.5,0.2,Iris-setosa
        4.8,3.4,1.6,0.2,Iris-setosa
        4.8,3.0,1.4,0.1,Iris-setosa
        4.3,3.0,1.1,0.1,Iris-setosa
        5.8,4.0,1.2,0.2,Iris-setosa
        5.7,4.4,1.5,0.4,Iris-setosa
        5.4,3.9,1.3,0.4,Iris-setosa
        5.1,3.5,1.4,0.3,Iris-setosa
        5.1,3.3,1.7,0.5,Iris-setosa
        4.8,3.4,1.9,0.2,Iris-setosa
        5.0,3.0,1.6,0.2,Iris-setosa
        5.2,3.5,1.5,0.2,Iris-setosa
        5.2,3.4,1.4,0.2,Iris-setosa
        4.7,3.2,1.6,0.2,Iris-setosa
        4.8,3.1,1.6,0.2,Iris-setosa
        5.4,3.4,1.5,0.4,Iris-setosa
        5.2,4.1,1.5,0.1,Iris-setosa
        5.5,4.2,1.4,0.2,Iris-setosa
        4.9,3.1,1.5,0.2,Iris-setosa
        5.0,3.2,1.2,0.2,Iris-setosa
        5.5,3.5,1.3,0.2,Iris-setosa
        4.9,3.6,1.4,0.1,Iris-setosa
        5.1,3.4,1.5,0.2,Iris-setosa
        5.0,3.5,1.3,0.3,Iris-setosa
        4.5,2.3,1.3,0.3,Iris-setosa
        4.6,3.2,1.4,0.2,Iris-setosa
        5.3,3.7,1.5,0.2,Iris-setosa
        5.0,3.3,1.4,0.2,Iris-setosa
        7.0,3.2,4.7,1.4,Iris-versicolor
        6.4,3.2,4.5,1.5,Iris-versicolor
        6.9,3.1,4.9,1.5,Iris-versicolor
        5.5,2.3,4.0,1.3,Iris-versicolor
        6.5,2.8,4.6,1.5,Iris-versicolor
        5.7,2.8,4.5,1.3,Iris-versicolor
        6.3,3.3,4.7,1.6,Iris-versicolor
        4.9,2.4,3.3,1.0,Iris-versicolor
        6.1,2.9,4.7,1.4,Iris-versicolor
        5.6,2.9,3.6,1.3,Iris-versicolor
        6.7,3.1,4.4,1.4,Iris-versicolor
        5.6,3.0,4.5,1.5,Iris-versicolor
        5.8,2.7,4.1,1.0,Iris-versicolor
        6.2,2.2,4.5,1.5,Iris-versicolor
        5.6,2.5,3.9,1.1,Iris-versicolor
        5.9,3.2,4.8,1.8,Iris-versicolor
        6.1,2.8,4.0,1.3,Iris-versicolor
        6.3,2.5,4.9,1.5,Iris-versicolor
        6.1,2.8,4.7,1.2,Iris-versicolor
        6.4,2.9,4.3,1.3,Iris-versicolor
        6.6,3.0,4.4,1.4,Iris-versicolor
        6.8,2.8,4.8,1.4,Iris-versicolor
        6.7,3.0,5.0,1.7,Iris-versicolor
        6.0,2.9,4.5,1.5,Iris-versicolor
        5.4,3.0,4.5,1.5,Iris-versicolor
        6.0,3.4,4.5,1.6,Iris-versicolor
        6.7,3.1,4.7,1.5,Iris-versicolor
        6.3,2.3,4.4,1.3,Iris-versicolor
        5.6,3.0,4.1,1.3,Iris-versicolor
        5.5,2.5,4.0,1.3,Iris-versicolor
        5.5,2.6,4.4,1.2,Iris-versicolor
        6.1,3.0,4.6,1.4,Iris-versicolor
        6.2,2.9,4.3,1.3,Iris-versicolor
        5.1,2.5,3.0,1.1,Iris-versicolor
        5.7,2.8,4.1,1.3,Iris-versicolor
        6.3,3.3,6.0,2.5,Iris-virginica
        5.8,2.7,5.1,1.9,Iris-virginica
        7.1,3.0,5.9,2.1,Iris-virginica
        6.7,2.5,5.8,1.8,Iris-virginica
        7.2,3.6,6.1,2.5,Iris-virginica
        6.5,3.2,5.1,2.0,Iris-virginica
        6.4,2.7,5.3,1.9,Iris-virginica
        6.8,3.0,5.5,2.1,Iris-virginica
        5.7,2.5,5.0,2.0,Iris-virginica
        6.0,2.2,5.0,1.5,Iris-virginica
        6.9,3.2,5.7,2.3,Iris-virginica
        5.6,2.8,4.9,2.0,Iris-virginica
        7.7,2.8,6.7,2.0,Iris-virginica
        6.3,2.7,4.9,1.8,Iris-virginica
        6.7,3.3,5.7,2.1,Iris-virginica
        7.2,3.2,6.0,1.8,Iris-virginica
        6.2,2.8,4.8,1.8,Iris-virginica
        6.1,3.0,4.9,1.8,Iris-virginica
        6.4,2.8,5.6,2.1,Iris-virginica
        7.2,3.0,5.8,1.6,Iris-virginica
        7.4,2.8,6.1,1.9,Iris-virginica
        7.9,3.8,6.4,2.0,Iris-virginica
        6.4,2.8,5.6,2.2,Iris-virginica
        6.3,2.8,5.1,1.5,Iris-virginica
        6.1,2.6,5.6,1.4,Iris-virginica
        7.7,3.0,6.1,2.3,Iris-virginica
        6.3,3.4,5.6,2.4,Iris-virginica
        6.4,3.1,5.5,1.8,Iris-virginica
        6.0,3.0,4.8,1.8,Iris-virginica
        6.9,3.1,5.4,2.1,Iris-virginica
        6.7,3.1,5.6,2.4,Iris-virginica
        6.9,3.1,5.1,2.3,Iris-virginica
        5.8,2.7,5.1,1.9,Iris-virginica
        6.2,3.4,5.4,2.3,Iris-virginica
        5.9,3.0,5.1,1.8,Iris-virginica
	*/
	double entradaTreinoBruto[][] = {
            {5.1,3.5,1.4,0.2},
            {4.6,3.1,1.5,0.2},
            {5.0,3.6,1.4,0.2},
            {4.6,3.4,1.4,0.3},
            {5.0,3.4,1.5,0.2},
            {4.4,2.9,1.4,0.2},
            {4.9,3.1,1.5,0.1},
            {5.4,3.7,1.5,0.2},
            {4.8,3.4,1.6,0.2},
            {4.8,3.0,1.4,0.1},
            {4.3,3.0,1.1,0.1},
            {5.8,4.0,1.2,0.2},
            {5.7,4.4,1.5,0.4},
            {5.4,3.9,1.3,0.4},
            {5.1,3.5,1.4,0.3},
            {5.1,3.3,1.7,0.5},
            {4.8,3.4,1.9,0.2},
            {5.0,3.0,1.6,0.2},
            {5.2,3.5,1.5,0.2},
            {5.2,3.4,1.4,0.2},
            {4.7,3.2,1.6,0.2},
            {4.8,3.1,1.6,0.2},
            {5.4,3.4,1.5,0.4},
            {5.2,4.1,1.5,0.1},
            {5.5,4.2,1.4,0.2},
            {4.9,3.1,1.5,0.2},
            {5.0,3.2,1.2,0.2},
            {5.5,3.5,1.3,0.2},
            {4.9,3.6,1.4,0.1},
            {5.1,3.4,1.5,0.2},
            {5.0,3.5,1.3,0.3},
            {4.5,2.3,1.3,0.3},
            {4.6,3.2,1.4,0.2},
            {5.3,3.7,1.5,0.2},
            {5.0,3.3,1.4,0.2},
            {7.0,3.2,4.7,1.4},
            {6.4,3.2,4.5,1.5},
            {6.9,3.1,4.9,1.5},
            {5.5,2.3,4.0,1.3},
            {6.5,2.8,4.6,1.5},
            {5.7,2.8,4.5,1.3},
            {6.3,3.3,4.7,1.6},
            {4.9,2.4,3.3,1.0},
            {6.1,2.9,4.7,1.4},
            {5.6,2.9,3.6,1.3},
            {6.7,3.1,4.4,1.4},
            {5.6,3.0,4.5,1.5},
            {5.8,2.7,4.1,1.0},
            {6.2,2.2,4.5,1.5},
            {5.6,2.5,3.9,1.1},
            {5.9,3.2,4.8,1.8},
            {6.1,2.8,4.0,1.3},
            {6.3,2.5,4.9,1.5},
            {6.1,2.8,4.7,1.2},
            {6.4,2.9,4.3,1.3},
            {6.6,3.0,4.4,1.4},
            {6.8,2.8,4.8,1.4},
            {6.7,3.0,5.0,1.7},
            {6.0,2.9,4.5,1.5},
            {5.4,3.0,4.5,1.5},
            {6.0,3.4,4.5,1.6},
            {6.7,3.1,4.7,1.5},
            {6.3,2.3,4.4,1.3},
            {5.6,3.0,4.1,1.3},
            {5.5,2.5,4.0,1.3},
            {5.5,2.6,4.4,1.2},
            {6.1,3.0,4.6,1.4},
            {6.2,2.9,4.3,1.3},
            {5.1,2.5,3.0,1.1},
            {5.7,2.8,4.1,1.3},
            {6.3,3.3,6.0,2.5},
            {5.8,2.7,5.1,1.9},
            {7.1,3.0,5.9,2.1},
            {6.7,2.5,5.8,1.8},
            {7.2,3.6,6.1,2.5},
            {6.5,3.2,5.1,2.0},
            {6.4,2.7,5.3,1.9},
            {6.8,3.0,5.5,2.1},
            {5.7,2.5,5.0,2.0},
            {6.0,2.2,5.0,1.5},
            {6.9,3.2,5.7,2.3},
            {5.6,2.8,4.9,2.0},
            {7.7,2.8,6.7,2.0},
            {6.3,2.7,4.9,1.8},
            {6.7,3.3,5.7,2.1},
            {7.2,3.2,6.0,1.8},
            {6.2,2.8,4.8,1.8},
            {6.1,3.0,4.9,1.8},
            {6.4,2.8,5.6,2.1},
            {7.2,3.0,5.8,1.6},
            {7.4,2.8,6.1,1.9},
            {7.9,3.8,6.4,2.0},
            {6.4,2.8,5.6,2.2},
            {6.3,2.8,5.1,1.5},
            {6.1,2.6,5.6,1.4},
            {7.7,3.0,6.1,2.3},
            {6.3,3.4,5.6,2.4},
            {6.4,3.1,5.5,1.8},
            {6.0,3.0,4.8,1.8},
            {6.9,3.1,5.4,2.1},
            {6.7,3.1,5.6,2.4},
            {6.9,3.1,5.1,2.3},
            {5.8,2.7,5.1,1.9},
            {6.2,3.4,5.4,2.3},
            {5.9,3.0,5.1,1.8}
        };
	double saidaTreino[][] = {
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0}
        };
        /*
        Dados de teste
        4.7,3.2,1.3,0.2,Iris-setosa
        5.0,3.4,1.6,0.4,Iris-setosa
        4.4,3.0,1.3,0.2,Iris-setosa
        4.9,3.0,1.4,0.2,Iris-setosa
        5.4,3.9,1.7,0.4,Iris-setosa
        5.7,3.8,1.7,0.3,Iris-setosa
        5.1,3.8,1.5,0.3,Iris-setosa
        5.4,3.4,1.7,0.2,Iris-setosa
        5.1,3.7,1.5,0.4,Iris-setosa
        4.6,3.6,1.0,0.2,Iris-setosa
        4.4,3.2,1.3,0.2,Iris-setosa
        5.0,3.5,1.6,0.6,Iris-setosa
        5.1,3.8,1.9,0.4,Iris-setosa
        4.8,3.0,1.4,0.3,Iris-setosa
        5.1,3.8,1.6,0.2,Iris-setosa
        6.6,2.9,4.6,1.3,Iris-versicolor
        5.2,2.7,3.9,1.4,Iris-versicolor
        5.0,2.0,3.5,1.0,Iris-versicolor
        5.9,3.0,4.2,1.5,Iris-versicolor
        6.0,2.2,4.0,1.0,Iris-versicolor
        5.8,2.6,4.0,1.2,Iris-versicolor
        5.0,2.3,3.3,1.0,Iris-versicolor
        5.6,2.7,4.2,1.3,Iris-versicolor
        5.7,3.0,4.2,1.2,Iris-versicolor
        5.7,2.9,4.2,1.3,Iris-versicolor
        5.7,2.6,3.5,1.0,Iris-versicolor
        5.5,2.4,3.8,1.1,Iris-versicolor
        5.5,2.4,3.7,1.0,Iris-versicolor
        5.8,2.7,3.9,1.2,Iris-versicolor
        6.0,2.7,5.1,1.6,Iris-versicolor
        6.8,3.2,5.9,2.3,Iris-virginica
        6.7,3.3,5.7,2.5,Iris-virginica
        6.7,3.0,5.2,2.3,Iris-virginica
        6.3,2.5,5.0,1.9,Iris-virginica
        6.5,3.0,5.2,2.0,Iris-virginica
        5.8,2.8,5.1,2.4,Iris-virginica
        6.4,3.2,5.3,2.3,Iris-virginica
        6.5,3.0,5.5,1.8,Iris-virginica
        7.7,3.8,6.7,2.2,Iris-virginica
        7.7,2.6,6.9,2.3,Iris-virginica
        6.3,2.9,5.6,1.8,Iris-virginica
        6.5,3.0,5.8,2.2,Iris-virginica
        7.6,3.0,6.6,2.1,Iris-virginica
        4.9,2.5,4.5,1.7,Iris-virginica
        7.3,2.9,6.3,1.8,Iris-virginica
        */
        double entradaTesteBruto[][] = {
            {4.7,3.2,1.3,0.2},
            {5.0,3.4,1.6,0.4},
            {4.4,3.0,1.3,0.2},
            {4.9,3.0,1.4,0.2},
            {5.4,3.9,1.7,0.4},
            {5.7,3.8,1.7,0.3},
            {5.1,3.8,1.5,0.3},
            {5.4,3.4,1.7,0.2},
            {5.1,3.7,1.5,0.4},
            {4.6,3.6,1.0,0.2},
            {4.4,3.2,1.3,0.2},
            {5.0,3.5,1.6,0.6},
            {5.1,3.8,1.9,0.4},
            {4.8,3.0,1.4,0.3},
            {5.1,3.8,1.6,0.2},
            {6.6,2.9,4.6,1.3},
            {5.2,2.7,3.9,1.4},
            {5.0,2.0,3.5,1.0},
            {5.9,3.0,4.2,1.5},
            {6.0,2.2,4.0,1.0},
            {5.8,2.6,4.0,1.2},
            {5.0,2.3,3.3,1.0},
            {5.6,2.7,4.2,1.3},
            {5.7,3.0,4.2,1.2},
            {5.7,2.9,4.2,1.3},
            {5.7,2.6,3.5,1.0},
            {5.5,2.4,3.8,1.1},
            {5.5,2.4,3.7,1.0},
            {5.8,2.7,3.9,1.2},
            {6.0,2.7,5.1,1.6},
            {6.8,3.2,5.9,2.3},
            {6.7,3.3,5.7,2.5},
            {6.7,3.0,5.2,2.3},
            {6.3,2.5,5.0,1.9},
            {6.5,3.0,5.2,2.0},
            {5.8,2.8,5.1,2.4},
            {6.4,3.2,5.3,2.3},
            {6.5,3.0,5.5,1.8},
            {7.7,3.8,6.7,2.2},
            {7.7,2.6,6.9,2.3},
            {6.3,2.9,5.6,1.8},
            {6.5,3.0,5.8,2.2},
            {7.6,3.0,6.6,2.1},
            {4.9,2.5,4.5,1.7},
            {7.3,2.9,6.3,1.8}
        };
        double saidaTeste[][] = {
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {1.0,-1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,1.0,-1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0},
            {-1.0,-1.0,1.0}
        };
        
        /** codificação das classes
         * {1, -1, -1}: Iris-setosa
         * {-1, 1, -1}: Iris-versicolor
         * {-1, -1, 1}: Iris-virginica
         **/
        
        // criando a rede neural
        BasicNetwork network = new BasicNetwork();
	network.addLayer(new BasicLayer(null,true,4));
	network.addLayer(new BasicLayer(new ActivationTANH(),true,6));
	network.addLayer(new BasicLayer(new ActivationTANH(),false,3));
	network.getStructure().finalizeStructure();
	network.reset();
        
        // normalizando os dados
        Normaliza norm = new Normaliza(entradaTreinoBruto,entradaTesteBruto, -1.0, 1.0);
        double entradaTreino[][] = norm.normalizar(entradaTreinoBruto);
        double entradaTeste[][] = norm.normalizar(entradaTesteBruto);
        
        // criando conjunto de treino e conjunto de teste
	MLDataSet trainingSet = new BasicMLDataSet(entradaTreino, saidaTreino);
	MLDataSet testSet = new BasicMLDataSet(entradaTeste, saidaTeste);
	// treinando a rede neural
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
	int epoch = 1;
        ArrayList<Double> erro = new ArrayList<>();
	do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Erro:" + train.getError());
            epoch++;
            erro.add(train.getError());
            // para se oscilar
            if(erro.size() > 5 && 
               erro.get(erro.size()-1) == erro.get(erro.size()-3) && 
               erro.get(erro.size()-1) == erro.get(erro.size()-5) ) break;
	} while(train.getError() > 0.009 && epoch <= 500000);
	// testando a rede neural
	System.out.println("Neural Network Results:");
	for(MLDataPair pair: testSet ) {
            final MLData output = network.compute(pair.getInput());
            System.out.println("{" +
                               norm.desnormalizar(pair.getInput().getData(0)) + "," + 
                               norm.desnormalizar(pair.getInput().getData(1)) + "," + 
                               norm.desnormalizar(pair.getInput().getData(2)) + "," + 
                               norm.desnormalizar(pair.getInput().getData(3)) + "} : " + 
                               RNSI.iris(output.getData(0),output.getData(1),output.getData(2))); 
	}
	Encog.getInstance().shutdown();
    }
    
    public static String iris(double x, double y, double z){
        int a, b, c;
        a = (int) Math.round(x);
        b = (int) Math.round(y);
        c = (int) Math.round(z);
        if(a==1) return "Iris-setosa";
        if(b==1) return "Iris-versicolor";
        if(c==1) return "Iris-virginica";
        return "Padrão desconhecido";
    }
    
}
