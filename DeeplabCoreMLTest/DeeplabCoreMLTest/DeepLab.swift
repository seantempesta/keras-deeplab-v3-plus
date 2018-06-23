//
//  DeepLab.swift
//  DeeplabCoreMLTest
//
//  Created by Sean Tempesta on 6/22/18.
//  Copyright © 2018 Sean Tempesta. All rights reserved.
//

import Foundation
import UIKit
import CoreML
import Accelerate

class DeepLab {
    public static let inputWidth = 512
    public static let inputHeight = 512
    public static let numClasses = 21
    
    
    let model = DeeplabMobilenet()
    
    public init() { }
    
    public func predict(image: CVPixelBuffer) throws  -> MultiArray<Double>?  {
        if let output = try? model.prediction(input_1: image) {
            return findArgmax(features: output.bilinear_upsampling_2)
        } else {
            return nil
        }
    }
    
    // The results are in a MLMultiArray with the values being [classes, height, width]
    // You'll need to find the largest value of each class for each pixel (an argmax)
    public func findArgmax(features: MLMultiArray) -> MultiArray<Double> {
        
        // NOTE: It turns out that accessing the elements in the multi-array as
        // `features[[cc, cy, cx] as [NSNumber]].floatValue` is kinda slow.
        // It's much faster to use direct memory access to the features.
        // Calculate the strides for each
        let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        let cStride = features.strides[0].intValue  // class stride
        let yStride = features.strides[1].intValue  // height stride
        let xStride = features.strides[2].intValue  // width stride
        
        // Convenience function to reach into the MLArray
        func offset(_ c: Int, _ x: Int, _ y: Int) -> Int {
            return c*cStride + y*yStride + x*xStride
        }
        
        // Create an array to store the argmax results in
        var argmaxResults = MultiArray<Double>(shape: [DeepLab.inputHeight, DeepLab.inputWidth])
        
        // This is slow, so I'm sure there's a faster way to do this
        for cy in 0..<DeepLab.inputHeight {
            for cx in 0..<DeepLab.inputWidth {
                
                var largestVal:Double = 0.0
                var largestClass:Double = 0.0
                
                for cc in 0..<DeepLab.numClasses {
                    let val = Double(featurePointer[offset(cc,cx,cy)])
                    if(val > largestVal) {
                        largestVal = val
                        largestClass = Double(cc)
                    }
                    
                }
                
                // Store the highest class for each pixel
                argmaxResults[cy,cx] = largestClass
            }
        }
        return argmaxResults
    }
}
