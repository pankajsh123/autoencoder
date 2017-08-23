#include<stdio.h>
#include<string.h>

int input(char * pattern, char * file)
{
	char c = '\0';
	int i = 0,flag = 0,temp = 0;
	
	while(1)
	{
		c = getchar();
		if((c == '\n' || c == ' ') && temp == 1)
			break;
		else if(c == ' ' && temp == 0)
			continue;
		else if(c == '\n' && temp == 0)
		{
			printf("Pattern not specified\n");
			return 0;
		}
		else 
		{
			pattern[i++] = c;
			temp = 1;
		}
	} 
	pattern[i] = '\0';
	if(c == '\n')
		return 0;
	
	i = 0;
	while(1)
	{
		c = getchar();
		if(c == '\n' && flag == 0) 
			return 0;
		else if(c == ' ' && flag == 0)
			continue;
		else if((c == ' ' || c == '\n') && flag == 1)
			break;
		else 
		{
			file[i++] = c;
			flag = 1;
		}
	} 
	file[i] = '\0';
	
	return 1;
}

int readFromFile(char * file, char * text)
{
	FILE * fp = fopen(file,"r");
	if(fp == NULL)
	{
		printf("No such file exists\n");
		return 0;
	}
	else
	{
		int i = 0;
		while(!feof(fp))
			text[i++] = fgetc(fp);
	}
	return 1;
}

int stringMatch(char * pattern, char * text)
{
	int i = 0,j = 0;
	while(1)
	{
		if(pattern[i] != '\0' && text[j] != '\0' && pattern[i] == text[j])
		{
			i++;
			j++;
		}
		else if(pattern[i] != '\0' && text[j] != '\0' && pattern[i] != text[j])
		{
			i = 0;
			j++;
		}
		else if(pattern[i] == '\0')
			return 1;
		else if(text[j] == '\0')
			return 0;
	}
}

int main()
{
	char grep[] = {'g','r','e','p'},c = '\0',pattern[1000],file[1000],text[10000];
	int i = 0;
	while(c != ' ')
	{
		c = getchar();
		if(c == grep[i] && i < 4)
			i++;
		else if(c != grep[i] && i < 4)
		{
			printf("Wrong input");
			return 0;
		}
	}
	
	int flag = input(pattern,file);
	if(flag)
	{
		readFromFile(file,text);
		int lineNum = 1,i = 0,j = 0,check;
		char line[1000];
		while(1)
		{
			while(text[i] != '\n' && text[i] != '\0')
				line[j++] =  text[i++];
			line[j] = '\0';
			j = 0;
			check = stringMatch(pattern,line);
			if(check)
				printf("Line Number is %d : %s\n",lineNum,line);
			if(text[i] == '\0')
				break;
			lineNum++;
			i++;
		}
	}
	else
	{
		while(1)
		{
			scanf("%s", text);
			int check = stringMatch(pattern,text);
			if(check)
				printf("pattern found\n");
		}
	}
	
	return 0;
}
