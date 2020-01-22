import numpy as np

class ReplaceNulls():
    def __init__(self):
        self.name = 'Replace Nulls'
        self.description = 'Replace NULL values in a raster with a user defined value.'



    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value':'Multiband Raster',
                'required': True,
                'displayName': 'Input Raster',
                'description': 'Input Raster'
            },
            {
                'name': 'fill_val',
                'dataType': 'numeric',
                'value': 65535,
                'required': True,
                'displayName': 'New NoData',
                'description': 'New NoData value to use'
            }
        ]

    def getConfiguration(self, **scalars):
        return {
            'inheritProperties': 1 | 4 | 8,       # inherit everything but NoData (2)
            'inputMask': True    #the input masks are made available in the pixelBlocks keyword
        }

    def updateRasterInfo(self, **kwargs):
        
        self.fill_val = kwargs['fill_val']
        bands = kwargs['raster_info']['bandCount']
        pixtype = kwargs['raster_info']['pixelType']

        kwargs['output_info']['bandCount'] = bands
        kwargs['output_info']['pixelType'] = pixtype
        kwargs['output_info']['noData'] = np.array(np.full(bands, fill_value=self.fill_val, dtype=pixtype))

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        pix_array = np.asarray(pixelBlocks['raster_pixels'])
        np.place(pix_array, pix_array==0, [self.fill_val])

        mask      = np.ones(pix_array.shape)
        pixelBlocks['output_mask'] = mask.astype('u1', copy = False)
        pixelBlocks['output_pixels'] = pix_array.astype(props['pixelType'], copy=True)


        return pixelBlocks


    