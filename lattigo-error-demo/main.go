package main

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/schemes/ckks"
)

func main() {
	// 1. 参数设置
	params, _ := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            13,
		LogQ:            []int{50, 40, 40},
		LogP:            []int{60},
		LogDefaultScale: 40,
	})

	ecd := ckks.NewEncoder(params)
	kgen := ckks.NewKeyGenerator(params)
	sk := kgen.GenSecretKey()
	enc := ckks.NewEncryptor(params, sk)
	dec := ckks.NewDecryptor(params, sk)
	eval := ckks.NewEvaluator(params, nil)

	maxSlots := params.MaxSlots()

	// 2. 准备明文向量 Pt = [1, 0, 0, ...]
	valuesPt := make([]complex128, maxSlots)
	valuesPt[0] = complex(1.0, 0)
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	ecd.Encode(valuesPt, pt)

	// ==========================================
	// 3. 手动注入噪声 (在多项式环级别)
	// 此时 pt.Value 存储的是 NTT 后的系数
	// 我们给第 0 个 RNS limb 的第 0 个系数加 1
	fmt.Println(">>> 正在明文多项式中注入大小为 1 的手动噪声...")
	pt.Value.Coeffs[0][0] = (pt.Value.Coeffs[0][0] + 1) % params.Q()[0]
	// ==========================================

	// 4. 准备密文向量 Ct = [1, 0, 0, ...]
	valuesCt := make([]complex128, maxSlots)
	valuesCt[0] = complex(1.0, 0)
	tempPt := ckks.NewPlaintext(params, params.MaxLevel())
	ecd.Encode(valuesCt, tempPt)
	ct, _ := enc.EncryptNew(tempPt)

	// 5. 执行同态乘法 (明密乘法)
	// 结果 res = (Pt + noise) * Ct
	resCt := eval.MulNew(ct, pt)

	// 6. 解密并解码
	resPt := dec.DecryptNew(resCt)
	resValues := make([]complex128, maxSlots)
	ecd.Decode(resPt, resValues)

	// 7. 打印结果
	fmt.Printf("\n--- 注入噪声后的结果 (前 5 个 Slot) ---\n")
	for i := 0; i < 5; i++ {
		fmt.Printf("Slot %d: %.10f\n", i, real(resValues[i]))
	}

	// 8. 分析误差扩散
	// 理论上，虽然你只改了一个多项式系数，但由于 Inverse NTT (Decoding)，
	// 这个“1”的噪声会均匀地扩散到所有的 Slot 中。
	errorCount := 0
	for _, v := range resValues {
		// 检查是否偏离了原始的 element-wise 预期 (Slot 0 是 1, 其他是 0)
		// 这里我们观察噪声造成的偏移
		if math.Abs(real(v)) > 0.1 && math.Abs(real(v)-1.0) > 0.1 {
			errorCount++
		}
	}
	fmt.Printf("\n总 Slots 数: %d", maxSlots)
	fmt.Printf("\n显著受噪声影响的 Slots 数: %d\n", errorCount)
}
