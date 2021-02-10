


def Final_Lithology_With_Comments():   #DB_lithology_With_Comments_Final_Export,minlong,maxlong,minlat,maxlat):
    '''
    Function Extracts data from tables dhgeologyattr,dhgeology,collar,clbody and attribute column lithology table from DB for the specified region.
    Also joins extraction of Comments attribute column with Comments attribute value .
    For Each row extracted, the from and to depth values are validated , generated fuzzywuzzy values for the lithology along with the score are printed .
    Input : 
        -minlong,maxlong,minlat,maxlat : Region of interest.
    Output:
        - csv file with the extracted data with fuzzywuzzy and score for lithology and comments.
    '''
    query = ''' SELECT m1.companyid, m1.collarid, m1.fromdepth, m1.todepth, m1.lith_attributecolumn, m1.lith_attributevalue, 
                m2.comments_attributecolumn, m2.comments_attributevalue 
                FROM 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS lith_attributecolumn, t1.attributevalue AS lith_attributevalue 
                 from public.dhgeologyattr t1 
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_lithology t5
                 on t1.attributecolumn = t5.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m1
                 FULL JOIN		 
                (select t1.dhgeologyid, t3.companyid, t2.collarid, t2.fromdepth, t2.todepth, t1.attributecolumn 
                 AS comments_attributecolumn, t1.attributevalue AS comments_attributevalue  
                 from public.dhgeologyattr t1 
                 inner join public.dhgeology t2 
                 on t1.dhgeologyid = t2.id 
                 inner join collar t3 
                 on t3.id = t2.collarid 
                 inner join clbody t4 
                 on t4.companyid = t3.companyid
                 inner join public.thesaurus_geology_comment t6
                 on t1.attributecolumn = t6.attributecolumn
                 WHERE(t3.longitude BETWEEN 115.5 AND 118) AND(t3.latitude BETWEEN -30.5 AND -27.5) 
                 ORDER BY t3.companyid ASC) m2 
                 on m1.dhgeologyid = m2.dhgeologyid'''
                 
                 
                 
        
    
    conn = psycopg2.connect(host = host_,port = port_,database = DB_,user = user_,password = pwd_)
    cur = conn.cursor()
    #Bounds=(minlong,maxlong,minlat,maxlat)  #query bounds 
    cur.execute(query)  #,Bounds)
    
    #print(cur)
    First_Filter_list = [list(elem) for elem in cur]
    
