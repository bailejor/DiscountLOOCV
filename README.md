The AIC_BIC.R file is used to produce AIC and BIC scores for each model and dataset using mixed-effects modeling. 
It produces the scores resulting in the following figures:
![AIC_Fig](https://github.com/bailejor/DiscountLOOCV/assets/4589448/6eb1641f-8c95-451e-b60e-f07f6843c123)
![BIC_Fig](https://github.com/bailejor/DiscountLOOCV/assets/4589448/277d0eeb-ae00-43b8-9dcc-d83d75fcf535)


The Discounting_NLMER_delayout file is used to perform LOOCV, holding out a delay at each iteration. It produces the scores resulting in the following figure:
![LOOCV_Fig](https://github.com/bailejor/DiscountLOOCV/assets/4589448/e12a82bd-862c-40ad-a0b9-29605cf6d594)


The Discounting_NLMER_delayout_resids.R file produces residuals at each fold of cross-validation and results in the following figure:
![LOOCV_Resids (-1 to 1)](https://github.com/bailejor/DiscountLOOCV/assets/4589448/5945b930-fcae-44d5-b967-6aa5a50d6a7f)

