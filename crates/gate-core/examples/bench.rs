use std::io::Cursor;
use weavetab_gate::{Gate, Input};

fn main() {
    let gate = Gate::default();

    if std::env::args().nth(1).as_deref() == Some("--batch") {
        let _ = gate.decide(Input::Text("warmup"), &["a", "b"]).unwrap();
        let samples: Vec<String> = (0..10_000)
            .map(|i| format!("ticket #{i}: payment webhook returned 502 on checkout"))
            .collect();
        let inputs: Vec<Input<'_>> = samples.iter().map(|s| Input::Text(s.as_str())).collect();
        let t0 = std::time::Instant::now();
        let res = gate
            .decide_batch(
                &inputs,
                &["production incident", "billing inquiry", "feature request"],
            )
            .unwrap();
        let ms = t0.elapsed().as_millis();
        println!(
            "batch 10000 unique: count={} total={}ms ({:.1} items/s) sample={:?}",
            res.len(),
            ms,
            10_000.0 / (ms as f64 / 1000.0).max(1e-6),
            res[0].choice
        );
        return;
    }

    if let Some(img_path) = std::env::args().nth(1) {
        let bytes = std::fs::read(&img_path).expect("failed to read image");
        // Warmup vision download/load
        let _ = gate.decide(Input::ImageBytes(&bytes), &["warmup", "other"]).unwrap();

        let res = gate
            .decide(
                Input::ImageBytes(&bytes),
                &[
                    "a Windows desktop taskbar with app icons and clock",
                    "a photo of a mountain landscape",
                    "a handwritten math equation",
                    "a cat sitting on a sofa",
                ],
            )
            .unwrap();
        println!("image ({} bytes): {:?}", bytes.len(), res);
        return;
    }

    // Warmup downloads/loads weights once before timing
    let _ = gate.decide(Input::Text("warmup"), &["a", "b"]).unwrap();

    let t1 = gate
        .decide(
            Input::Text("payment webhook returned 502 bad gateway on checkout"),
            &["production incident", "billing inquiry", "feature request"],
        )
        .unwrap();
    println!(
        "text uncached: choice={:?} conf={:.3} latency={}ms",
        t1.choice, t1.confidence, t1.latency_ms
    );

    let eval = [
        ("database connection pool exhausted in us-east-1", "production incident"),
        ("please send a copy of my invoice for last month", "billing inquiry"),
        ("double charged on my visa card yesterday", "billing inquiry"),
        ("can you add dark mode to the settings screen?", "feature request"),
        ("api returning 500 internal server error for all users", "production incident"),
    ];
    let eval_inputs: Vec<Input<'_>> = eval.iter().map(|(t, _)| Input::Text(t)).collect();
    let eval_res = gate
        .decide_batch(
            &eval_inputs,
            &["production incident", "billing inquiry", "feature request"],
        )
        .unwrap();
    let ok = eval_res
        .iter()
        .zip(eval.iter())
        .filter(|(r, (_, y))| r.choice == *y)
        .count();
    println!("eval accuracy: {ok}/{} ({:?})", eval.len(), eval_res.iter().map(|r| r.choice.as_str()).collect::<Vec<_>>());

    let t2 = gate
        .decide(
            Input::Text("payment webhook returned 502 bad gateway on checkout"),
            &["production incident", "billing inquiry", "feature request"],
        )
        .unwrap();
    println!(
        "text cached:   choice={:?} cached={} latency={}ms",
        t2.choice, t2.cached, t2.latency_ms
    );

    let red = image::RgbImage::from_pixel(224, 224, image::Rgb([230, 20, 20]));
    let mut png1 = Cursor::new(Vec::new());
    red.write_to(&mut png1, image::ImageFormat::Png).unwrap();
    let _ = gate
        .decide(Input::ImageBytes(png1.get_ref()), &["red", "blue"])
        .unwrap();

    let blue = image::RgbImage::from_pixel(224, 224, image::Rgb([15, 70, 235]));
    let mut png2 = Cursor::new(Vec::new());
    blue.write_to(&mut png2, image::ImageFormat::Png).unwrap();

    let v1 = gate
        .decide(
            Input::ImageBytes(png2.get_ref()),
            &["a bright blue color", "a bright red color", "a green forest"],
        )
        .unwrap();
    println!(
        "vision uncached: choice={:?} conf={:.3} latency={}ms",
        v1.choice, v1.confidence, v1.latency_ms
    );

    let v2 = gate
        .decide(
            Input::ImageBytes(png2.get_ref()),
            &["a bright blue color", "a bright red color", "a green forest"],
        )
        .unwrap();
    println!(
        "vision cached:   choice={:?} cached={} latency={}ms",
        v2.choice, v2.cached, v2.latency_ms
    );
}
