
import Foundation
import CoreML
import Accelerate


@objc(BilinearUpsampling) class BilinearUpsampling: NSObject, MLCustomLayer {
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws
        -> [[NSNumber]] {
            print(#function, inputShapes)
            return [[234,234,234,1,1]]
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        print(#function, inputs.count, outputs.count)
    }
}
