package com.example.slt_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
Button butn1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        butn1 = findViewById(R.id.btn1);


        butn1.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v){
                Intent relation = new Intent(getApplicationContext(),ModelAct.class);
            }
        });
    }
}