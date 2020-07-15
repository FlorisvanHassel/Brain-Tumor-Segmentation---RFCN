#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def change_to_3D_MRI(y_pred):
    MRI_indexes = np.linspace(0, int(len(y_pred)), num = int(len(y_pred)/41)+1 )[:-1]
    data_3D_mask = []
    for i in MRI_indexes:
        temp_3D = y_pred[int(i):(int(i)+41)]
        data_3D_mask.append(temp_3D)
    return np.asarray(data_3D_mask)


# In[ ]:


def calculate_dice(y_pred, y_true):
    y_pred = np.round(y_pred)
    tp = np.sum(np.where(y_pred + y_true == 2, 1, 0))
    fp_fn = np.sum(np.where(y_pred + y_true == 1, 1, 0))
    return (2*tp)/((2*tp) + fp_fn)


# In[ ]:


def calculate_metrics(y_pred, y_true):
    y_pred = np.round(y_pred).flatten().astype('int')
    y_true = y_true.flatten()

    confusion_matrix = np.zeros((2,2))
    for i, j in zip(y_pred, y_true):
        confusion_matrix[i][j] += 1

    sensitivity = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])
    specificity = confusion_matrix[0][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])
    return sensitivity, specificity


# In[ ]:


def calculate_stats_3D_data(y_pred, y_true, labels):
    t_count = 0
    t_dice = 0
    t_sen = 0
    t_spe = 0

    h_count = 0
    HGG_dice = 0
    HGG_sen = 0
    HGG_spe = 0

    l_count = 0
    LGG_dice = 0
    LGG_sen = 0
    LGG_spe = 0

    length_data = np.arange(len(y_pred))

    for i, j in zip(length_data, labels):
        test_dice = calculate_dice(y_pred[i], y_true[i])
        test_sen, test_spe = calculate_metrics(y_pred[i], y_true[i])
        t_count += 1
        t_dice += test_dice
        t_sen += test_sen
        t_spe += test_spe

    if j == 0:
        h_count += 1
        HGG_dice += test_dice
        HGG_sen += test_sen
        HGG_spe += test_spe

    if j == 1:
        l_count += 1
        LGG_dice += test_dice
        LGG_sen += test_sen
        LGG_spe += test_spe

    result_data = {'total': [t_dice/t_count, t_sen/t_count, t_spe/t_count], 'HGG': [HGG_dice/h_count, HGG_sen/h_count, HGG_spe/h_count], 'LGG': [LGG_dice/l_count, LGG_sen/l_count, LGG_spe/l_count]}
    result_dic = pd.DataFrame(result_data)
    result_dic = result_dic.rename(index={0:'Dice score', 1: 'Sensitivity', 2: 'Specificity'})

    return result_dic


# In[ ]:





# In[ ]:





# In[ ]:




