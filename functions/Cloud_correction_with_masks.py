import numpy as np

def comboRules(r1band, r2band, change):
    newBand = np.where(change, r2band, r1band)
    return newBand

class Cloud_correction_with_masks():
    def __init__(self):
        self.name = "CCM Function"
        self.description = "Takes a composite raster composed of 2 4-band images (RGB and NIR) \
                            and 1 single-band mask raster (so - 9 bands altogether) and replaces the values \
                            in all bands from the first image that fall within the mask (value = 1) with \
                            values from the second image, and outputs the 'corrected' 4-band image."
        self.applyScaling = False
        self.applyColormap = False

    # getParameterInfo() describes all raster and scalar inputs to the raster function.
    def getParameterInfo(self):
        return [{
                'name': 'rasters',
                'dataType': 'rasters',
                'value': None,
                'required': True,
                'displayName': "Rasters",
                'description': "The 9-band composite image"
        }]

    # getConfiguration() helps define attributes that configures how input rasters are read and the output raster constructed.
    def getConfiguration(self, **scalars):
        return {
          #'extractBands': # we want all the bands, and just have to assume they're in the right order.
          'compositeRasters': True,             # output is multi-band.
          'inheritProperties': 4 | 8,           # inherit spatial dimensions and resampling type
          'invalidateProperties': 2 | 4 | 8,    # reset any statistics and histogram that might be held by the parent dataset (because this function modifies pixel values).
          'inputMask': False                    # Don't need NoData mask in .updatePixels().
        }


    # updateRasterInfo() enables you to define the location and dimensions of the output raster.
    def updateRasterInfo(self, **kwargs): #The keyword argument kwargs contains all user-specified scalar values and information associated with each input rasters.
        pixelType = 'u2' # unsigned integer 16 bit

        kwargs['output_info']['bandCount'] = 4            # output is a 4-band raster
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['pixelType'] = pixelType    # bit-depth of the output raster
        kwargs['output_info']['noData'] = np.array([65535, 65535, 65535, 65535], 'u2')
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
        mask = np.array(inBlock[8], dtype='u1')

        change = np.where(np.equal(mask, 1), True, False)

        bR_outBlock = comboRules(r1_bR, r2_bR, change)
        bG_outBlock = comboRules(r1_bG, r2_bG, change)
        bB_outBlock = comboRules(r1_bB, r2_bB, change)
        bNIR_outBlock = comboRules(r1_bNIR, r2_bNIR, change)

        catBlock = np.concatenate((bR_outBlock, bG_outBlock, bB_outBlock, bNIR_outBlock), axis=0)
        x = shape[2]
        y = shape[1]
        outBlock = np.reshape(catBlock, (4, y, x))

        if self.applyScaling:
            outBlock = (outBlock * 100.0) + 100.0                  # apply a scale and offset, if needed.

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
