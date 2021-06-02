package com.example.slt_app;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class ModelAct extends AppCompatActivity {
    Button butn2;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.model_act);
        butn2 = findViewById(R.id.btn_model);

        butn2.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v){
                Intent relation = new Intent(getApplicationContext(),CamAct.class);
            }
        });
    }
}
