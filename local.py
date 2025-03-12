def triangulate_radar_position(local_target1, local_target2, 
                               global_target1=(406.5, 399.0), 
                               global_target2=(406.5, 123.5)):


    x1_local, y1_local = local_target1
    x2_local, y2_local = local_target2

    x1_global, y1_global = global_target1  
    x2_global, y2_global = global_target2  


    if y1_local >= y2_local:

        matched_local1 = (x1_local, y1_local)
        matched_global1 = (x1_global, y1_global)


        matched_local2 = (x2_local, y2_local )
        matched_global2 = (x2_global, y2_global)
    else:

        matched_local1 = (x2_local, y2_local)
        matched_global1 = (x1_global, y1_global)


        matched_local2 = (x1_local, y1_local)
        matched_global2 = (x2_global, y2_global)


    X1 = matched_global1[0] - matched_local1[0]
    Y1 = matched_global1[1] - matched_local1[1]

    X2 = matched_global2[0] - matched_local2[0]
    Y2 = matched_global2[1] - matched_local2[1]


    X = (X1 + X2) / 2.0
    Y = (Y1 + Y2) / 2.0

    return X, Y
