#include<stdio.h>
int main(void){
	FILE *fp,*fp2;
	char buffer[3];
	char ch;
	int count=0;
	char* filename="demo2.txt";
	if((fp=fopen(filename,"r+"))==NULL){
		printf("���ļ�����\n");
		fclose(fp);
		return 0; 
	}
	if((fp2=fopen("output2.txt","w+"))==NULL){//����ļ� 
		printf("���ļ�����\n");
		fclose(fp2);
		return 0; 
	}
	ch=fgetc(fp);
	while(ch!=EOF){
		buffer[0]=ch;
		fputc(ch,fp2);
		if(ch>127||ch<0){//�ַ�����ASCII�뷶Χ�ڣ���ȡ����Byte�ĺ��� 
			ch=fgetc(fp);
			buffer[1]=ch;
			fputc(ch,fp2);
			buffer[2]='\0';
			printf("%s ",buffer);
		}
		else
			printf("%c ",buffer[0]);//�ַ���ASCII�뷶Χ�� ����ȡһ��Byte���ַ� 
		fputc(' ',fp2);	
		count++;
		ch=fgetc(fp);
	}
	printf("%d\n",count); 
	fprintf(fp2,"%d",count);
	fclose(fp);
	return 1;
} 