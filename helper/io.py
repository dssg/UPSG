import csv

def print_Matrix_row_col( M, L, L_2,):
    """single line description
    Parameters
    ----------
    L : List 
       Description 
    M : List 
       Description
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    row_format ="{:>15}" * (len(L_2) + 1)
    print row_format.format("", *L_2)
    for team, row in zip(L, M):
        print row_format.format(team, *row)
    return None

    
    

def open_csv(loc):
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    with open(loc, 'rb') as f:
        reader = csv.reader(f)
        data= list(reader)
    return data
    
def save_csv(data, destination):
    with open(destination, 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)
    return



def convert_fixed_width_list_to_CSV_list(data, list_of_widths):
    s = "s".join([str(s) for s in list_of_widths])
    s= s + 's'
    out = []
    for x in data:
        out.append(struct.unpack(s, x[0]))
    return out