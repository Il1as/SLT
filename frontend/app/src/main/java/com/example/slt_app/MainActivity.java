package com.example.slt_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
Button button1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        button1 = findViewById(R.id.btn1);
        button1.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v){
                Intent relation = new Intent(getApplicationContext(),ModelAct.class);
                startActivity(relation);
                Toast.makeText(getApplicationContext(),"hello",Toast.LENGTH_LONG).show();
            }
        });
    }
}