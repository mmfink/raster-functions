import numpy as np

def comboRules(r1band, r2band, test1, test2, test3):
    newBand = np.where(np.logical_and(test1,test2), np.maximum(r1band, r2band), np.where(np.logical_and(test1,test3), np.minimum(r1band,r2band), (r1band + r2band) / 2))
    return newBand

class Cloud_correction_and_averaging():
    def __init__(self):
        self.name = "CCA Function"
        self.description = "This function takes 2 4-band images (RGB & NIR) of the same area from two different points in time and attempts to correct for clouds and other variability such as time of day. The rasters should be from very similar times of year."
        self.applyScaling = False
        self.applyColormap = False

    # getParameterInfo() describes all raster and scalar inputs to the raster function.
    def getParameterInfo(self):
        return [
            {
                'name': 'r1',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Raster 1",
                'description': "Must be a 4-band raster (RGB & NIR)"
            },
            {
                'name': 'r2',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Raster 2",
                'description': "Must be a 4-band raster (RGB & NIR) in same order as the first raster"
            },
            {
                'name': 'R',
                'dataType': 'numeric',
                'value': 1,
                'required': True,
                'displayName': "Red Band Index",
                'description': "The index of the red band. The first band has index 1."
            },
            {
                'name': 'G',
                'dataType': 'numeric',
                'value': 2,
                'required': True,
                'displayName': "Green Band Index",
                'description': "The index of the green band. The first band has index 1."
            },
            {
                'name': 'B',
                'dataType': 'numeric',
                'value': 3,
                'required': True,
                'displayName': "Blue Band Index",
                'description': "The index of the blue band. The first band has index 1."
            },
            {
                'name': 'NIR',
                'dataType': 'numeric',
                'value': 4,
                'required': True,
                'displayName': "Near Infrared Band Index",
                'description': "The index of the near-infrared band. The first band has index 1."
            },
        ]

    # getConfiguration() helps define attributes that configures how input rasters are read and the output raster constructed.
    def getConfiguration(self, **scalars):
        bR = int(scalars.get('R', 1))   #Indexes are actually 0-based so have to correct.
        bG = int(scalars.get('G', 2))
        bB = int(scalars.get('B', 3))
        bNIR = int(scalars.get('NIR', 4))
        
        return {
          'extractBands': (bR - 1, bG - 1, bB - 1, bNIR - 1),   # we want all the bands, but since they might not be in expected order, we extract them anyway. 
          'compositeRasters': True,             # output is multi-band.
          'inheritProperties': 4 | 8,           # inherit all but the pixel type and NoData from the input raster
          'invalidateProperties': 2 | 4 | 8,    # reset any statistics and histogram that might be held by the parent dataset (because this function modifies pixel values).
          'inputMask': False                    # Don't need input raster mask in .updatePixels().
        }


    # updateRasterInfo() enables you to define the location and dimensions of the output raster.
    def updateRasterInfo(self, **kwargs): #The keyword argument kwargs contains all user-specified scalar values and information associated with each input rasters. 
        pixelType = 'u2' # unsigned integer 16 bit

        kwargs['output_info']['bandCount'] = 4            # output is a 4-band raster
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['pixelType'] = pixelType    # bit-depth of the outgoing CCA raster
        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        pixvals = pixelBlocks.values()

        inBlock = np.asarray(pixvals[0])
        r1_bR = np.array(inBlock[0], dtype='u2')
        r1_bG = np.array(inBlock[1], dtype='u2')
        r1_bB = np.array(inBlock[2], dtype='u2')
        r1_bNIR = np.array(inBlock[3], dtype='u2')
        r2_bR = np.array(inBlock[4], dtype='u2')
        r2_bG = np.array(inBlock[5], dtype='u2')
        r2_bB = np.array(inBlock[6], dtype='u2')
        r2_bNIR = np.array(inBlock[7], dtype='u2')

        np.seterr(divide='ignore')
        # the treatment of all bands must match the outcomes from the Red band
        test1 = np.where(np.greater(np.fabs(r1_bR - r2_bR), 1000), True, False)  # there is a large difference
        # a likely cloud shadow in one band:
        test2 = np.where(np.logical_and(np.logical_and(np.less(r1_bR, 3000), np.less(r2_bR, 3000)), np.logical_xor(np.less(r1_bR, 500), np.less(r2_bR, 500))), True, False)
        # a likely cloud in one band:
        test3 = np.where(np.logical_xor(np.greater(r1_bR, 3000), np.greater(r2_bR, 3000)), True, False)

        bR_outBlock = comboRules(r1_bR, r2_bR, test1, test2, test3)
        bG_outBlock = comboRules(r1_bG, r2_bG, test1, test2, test3)
        bB_outBlock = comboRules(r1_bB, r2_bB, test1, test2, test3)
        bNIR_outBlock = comboRules(r1_bNIR, r2_bNIR, test1, test2, test3)

        catBlock = np.concatenate((bR_outBlock, bG_outBlock, bB_outBlock, bNIR_outBlock), axis=0)
        x = shape[2]
        y = shape[1]
        outBlock = np.reshape(catBlock, (4, y, x))

        if self.applyScaling:
            outBlock = (outBlock * 100.0) + 100.0                  # apply a scale and offset to the the CCA, if needed.

        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'])
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:
            keyMetadata['datatype'] = 'Processed'               # outgoing raster is now 'Processed' 
        elif bandIndex == 0:
            keyMetadata['bandname'] = 'Red'
        elif bandIndex == 1:
            keyMetadata['bandname'] = 'Green'
        elif bandIndex == 2:
            keyMetadata['bandname'] = 'Blue'
        elif bandIndex == 3:
            keyMetadata['bandname'] = 'Near-IR'
        return keyMetadata
