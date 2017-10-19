import sys

def convert_id_to_int(input_file, output_file):

    out=open(output_file,"w", encoding="utf-8")    
    f=open(input_file, "r", encoding="utf-8")
    lines=f.readlines()

    for line in lines:
        parts=line.split("\t")
        if len(parts)==4:
     
            id=parts[0]
            id_int = id.split("-")[-1]
            tweet=parts[1]
            emotion=parts[2]
            score=parts[3].strip() 
            score = score if score != "NONE" else "?"
            
            out_line=id_int+'\t'+tweet+'\t'+emotion+'\t'+score+'\n'
            out.write(out_line)
        else:
            print("Wrong format")

    

    f.close()  
    out.close()  

def main(argv):
    input_file=argv[0]
    output_file=argv[1]
    convert_id_to_int(input_file,output_file)
   
        
if __name__ == "__main__":
    main(sys.argv[1:])    
    