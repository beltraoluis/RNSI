/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rnsi;

import java.text.DecimalFormat;

/**
 *
 * @author beltraoluis
 */
public class Normaliza {

    private double min;
    private double max;
    private double nmin;
    private double nmax;

    public Normaliza(double[][] mat,double  nmin,double nmax) {
        this.nmin = nmin;
        this.nmax = nmax;
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;
        for(double[] l: mat){
            for(double c: l){
                if(c<this.min) this.min = c;
                if(c>this.max) this.max = c;
            }
        }
    }
    
    public Normaliza(double[][] mat, double[][] mat2,double  nmin,double nmax) {
        this.nmin = nmin;
        this.nmax = nmax;
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;
        for(double[] l: mat){
            for(double c: l){
                if(c<this.min) this.min = c;
                if(c>this.max) this.max = c;
            }
        }
        for(double[] l: mat2){
            for(double c: l){
                if(c<this.min) this.min = c;
                if(c>this.max) this.max = c;
            }
        }
    }

    public void setMin(double min) {
        this.min = min;
    }

    public void setMax(double max) {
        this.max = max;
    }

    public double getMin() {
        return min;
    }

    public double getMax() {
        return max;
    }
    
    public double[][] normalizar(double[][] mat){
        int linha = 0;
        int coluna = 0;
        for(double[] l: mat){
            linha++;
            for(double c: l){
                if(linha == 1) coluna++;
            }
        }
        double saida[][] = new double[linha][coluna];
        for(int i = 0; i<linha; i++){
            for(int j = 0; j<coluna; j++){
                saida[i][j] = ((mat[i][j]-this.min)*(nmax-nmin)/(this.max-this.min))+nmin;
            }
        }
        return saida;
    }
    
    double desnormalizar(double x){
        double n = Math.round(10*((min-max)*x-nmax*min+max*nmin)/(nmin-nmax));
        return n/10;
    }
    
}
