/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning_coursework;

import weka.core.Instance;

/**
 *
 * @author yqm15fqu
 * @param <Instance>
 * @param <distance>
 */
public class InstanceContainer implements Comparable<InstanceContainer>{

    private Instance instance;
    private double distance;

    public InstanceContainer(Instance inst, double dist) {
        this.instance = inst;
        this.distance = dist;
    }
    
    public Instance getInstance(){
        return this.instance;
    }
    
    public double getDistance(){
        return this.distance;
    }

    @Override
    public int compareTo(InstanceContainer inst) {
            if(this.distance < inst.distance){
                return -1;
            }else if(inst.distance < this.distance){
                return 1;
            }            
            return 0;
    }




    
    
    
}
