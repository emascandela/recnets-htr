# mkdir -p data ;
cd /root/data ;

if [ ! -f downloaded ] ; then 
pip install gdown;
python -m gdown 1Bi_iaV42CcUz-_QaruOKBxi1rEiihKRy -O IAM.tgz ;
python -m gdown 1nGX3dsiHaBkekSkzaWJPyluHJeiRdQmr -O PARZIVAL.tgz ;
python -m gdown 1g3Toaz22i5Jtap-ERxnXJQ29XETLtpeq -O SAINT_GALL.tgz ;
python -m gdown 1QlMftPVmDYQobO7RGiiM4L3HmItX8ijW -O WASHINGTON.tgz ;

tar xf SAINT_GALL.tgz ;
tar xf PARZIVAL.tgz ;
tar xf IAM.tgz ;
tar xf WASHINGTON.tgz ;
cp -rf IAM IAM_S ;
cd - ;
python resize_iam.py ;

touch downloaded;
fi
