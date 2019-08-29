/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package multivariate_timeseriesweka.classifiers;

import multivariate_timeseriesweka.measures.DTW_I;
import static utilities.InstanceTools.findMinDistance;
import utilities.generic_storage.Pair;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class NN_DTW_I extends AbstractClassifier{
    
    Instances train;
    DTW_I I;
    public NN_DTW_I(){
        I = new DTW_I();
    }
    
    public void setR(double r){
        I.setR(r);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train = data;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception{
        Pair<Instance, Double> minD = findMinDistance(train, instance, I);
        return minD.var1.classValue();
    }
}
