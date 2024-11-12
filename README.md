Main Article: https://www.mdpi.com/2076-3417/14/17/7564

Code Article: https://github.com/karryxz/Modified-model/blob/main/modified_mobilenetv3.py

Dataset Article: https://ieeexplore.ieee.org/abstract/document/7312934/authors#authors

Dataset Link: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz

#### Filename Image Format

`<BIOPSY_PROCEDURE>_<TUMOR_CLASS>_<TUMOR_TYPE>_<YEAR>-<SLIDE_ID>-<MAGNIFICATION>-<SEQ>.png`

```
BIOPSY_PROCEDURE: SOB
TUMOR_CLASS: B|M
TUMOR_TYPE: BENIGN_TYPE: A|F|PT|TA Or MALIGNANT_TYPE: DC|LC|MC|PC
YEAR: DIGIT
SLIDE_ID: NUMBER,SECTION
SEQ: NUMBER
MAGNIFICATION: 40|100|200|400
```
Where:
   - SOB = Surgical Open Biopsy
   - B = Benign
       - A = Adenosis
   	   - F = Fibroadenoma
       - TA = Tubular Adenoma
       - PT = Phyllodes Tumor
   - M = Malignant
	   - DC = Ductal Carcinoma
       - LC = Lobular Carcinoma
       - MC = Mucinous Carcinoma 
       - PC = Papillary Carcinoma