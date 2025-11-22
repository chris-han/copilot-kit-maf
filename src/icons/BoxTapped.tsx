import { PackageOpen as LucidePackageOpen, LucideProps } from 'lucide-react';

const BoxTapped = ({ className, ...props }: LucideProps) => {
  return <LucidePackageOpen className={className} {...props} />;
};

export default BoxTapped;